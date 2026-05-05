"""
CRIME CLASSIFICATION - CSIS355 Data Mining Project (UPDATED: "NE LAŽIRAJ VIOLENT" + FIX LEAKAGE)
Goal:
- Stop data leakage (pipelines fit only on TRAIN)
- HARD STOP leakage: drop columns that directly/indirectly encode the target (Crm Cd family)
- Do NOT "fake" violent => prioritize HIGH PRECISION for class=1 (Violent)
- Threshold tuning is done on VAL only (no test leakage)
- Final report is on TEST only

Key fixes vs your last code:
1) Adds explicit LEAKAGE_COLS: drops Crm Cd / Crm Cd 1-4 (and other target-derived fields) so target can't leak.
2) Fixes IG feature names (occ_day_of_week / rpt_day_of_week instead of day_of_week).
3) Keeps manual theory clearly as an educational demo on TRAIN only (no leakage).
"""

from __future__ import annotations

import gc
import time
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score, make_scorer,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Config:
    DATA_PATH: str = "clean.csv"
    TARGET_COLUMN: str = "is_violent_crime"

    SAMPLE_SIZE: int = 300_000
    RANDOM_STATE: int = 42

    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.15

    RESULTS_DIR: str = "results"
    FIGURE_DPI: int = 150

    # Columns that are either IDs, text fields, or high-cardinality and not useful for classic models here
    HIGH_CARDINALITY_COLS: Tuple[str, ...] = (
        "DR_NO",
        "LOCATION",
        "Cross Street",
        "Mocodes",
        "Crm Cd Desc",
        "Weapon Desc",
        "Premis Desc",
        "Status Desc",
        "AREA NAME",
        "Date Rptd",
        "DATE OCC",
    )

    # HARD leakage drops: target is derived from Crm Cd => Crm Cd family MUST NOT be in features
    # (drop even if you *think* you already removed them in preprocessing)
    LEAKAGE_COLS: Tuple[str, ...] = (
        "Crm Cd",
        "Crm Cd 1",
        "Crm Cd 2",
        "Crm Cd 3",
        "Crm Cd 4",
    )

    # manual theory demo candidates (must exist after dropping)
    IG_FEATURES_TO_ANALYZE: Tuple[str, ...] = (
        "hour", "AREA", "Vict Age", "occ_day_of_week", "rpt_day_of_week", "is_armed"
    )

    # PEBLS uses discrete/symbolic features
    PEBLS_FEATURES: Tuple[str, ...] = ("Vict Sex", "time_period", "season")

    # Decision Tree
    DT_MAX_DEPTH: int = 10
    DT_MIN_SAMPLES_SPLIT: int = 200
    DT_MIN_SAMPLES_LEAF: int = 100
    PRUNING_GRID_SIZE: int = 12

    MAX_CATEGORIES_ONEHOT: int = 10

    # KNN
    KNN_K: int = 7

    # CV
    CV_FOLDS: int = 5
    CV_SAMPLE_SIZE: int = 60_000

    # PEBLS controls
    PEBLS_MAX_EXEMPLARS: int = 25_000
    PEBLS_EPOCHS: int = 2
    PEBLS_TUNE_EXEMPLARS: int = 3_000
    PEBLS_TUNE_CANDIDATES: int = 2_000
    PEBLS_PRED_MAX_CANDIDATES: int = 5_000
    PEBLS_PRED_FALLBACK_CANDIDATES: int = 5_000

    # IMPORTANT: "NE LAŽIRAJ VIOLENT"
    MIN_PRECISION_VIOLENT_VAL: float = 0.55
    THRESHOLDS: Tuple[float, ...] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))
    FBETA: float = 0.5


# ==============================================================================
# CLONE-SAFE CUSTOM TRANSFORMERS
# ==============================================================================
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop  # clone-safe: never modify

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = list(self.cols_to_drop) if self.cols_to_drop is not None else []
            return X.drop(columns=[c for c in cols if c in X.columns], errors="ignore")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            return np.array([], dtype=object)
        cols = list(self.cols_to_drop) if self.cols_to_drop is not None else []
        return np.array([c for c in input_features if c not in cols], dtype=object)


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols  # clone-safe

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SelectColumns expects a pandas DataFrame.")
        wanted = list(self.cols) if self.cols is not None else []
        cols = [c for c in wanted if c in X.columns]
        if not cols:
            raise ValueError("SelectColumns: none of requested columns exist in X.")
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        wanted = list(self.cols) if self.cols is not None else []
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            return np.array(wanted, dtype=object)
        return np.array([c for c in wanted if c in input_features], dtype=object)


class ImputeMostFrequentDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imp = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ImputeMostFrequentDF expects a pandas DataFrame.")
        self.columns_ = list(X.columns)
        self.imp.fit(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ImputeMostFrequentDF expects a pandas DataFrame.")
        arr = self.imp.transform(X)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(getattr(self, "columns_", []), dtype=object)


class CastToStrDF(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CastToStrDF expects a pandas DataFrame.")
        out = X.copy()
        for c in out.columns:
            out[c] = out[c].astype(str).fillna("__UNK__")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(getattr(self, "columns_", []), dtype=object)


class TransformerMixinWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)


# ==============================================================================
# UTILITIES
# ==============================================================================
def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{title:^78}")
    print("=" * 78)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: str, dpi: int = 150) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=15)
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]:d}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def binary_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / max((tn + fp + fn + tp), 1)
    prec1 = tp / max((tp + fp), 1)
    rec1 = tp / max((tp + fn), 1)
    f11 = 0.0 if (prec1 + rec1) == 0 else 2 * prec1 * rec1 / (prec1 + rec1)
    rec0 = tn / max((tn + fp), 1)  # TNR
    bal_acc = 0.5 * (rec0 + rec1)
    fpr = fp / max((fp + tn), 1)
    return {
        "Accuracy": acc,
        "Precision_1": prec1,
        "Recall_1": rec1,
        "F1_1": f11,
        "Recall_0": rec0,
        "Balanced_Acc": bal_acc,
        "FPR_fake_violent": fpr,
        "TP": float(tp), "FP": float(fp), "TN": float(tn), "FN": float(fn)
    }


def fbeta_from_pr(prec: float, rec: float, beta: float) -> float:
    b2 = beta * beta
    denom = (b2 * prec + rec)
    if denom <= 0:
        return 0.0
    return (1 + b2) * prec * rec / denom


def tune_threshold_for_precision(
    y_true: np.ndarray,
    proba_1: np.ndarray,
    thresholds: List[float],
    beta: float,
    min_precision: float
) -> Dict[str, Any]:
    best = None
    best_any = None

    for thr in thresholds:
        y_pred = (proba_1 >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        m = binary_metrics_from_cm(cm)
        fbeta = fbeta_from_pr(m["Precision_1"], m["Recall_1"], beta)

        row = {
            "threshold": float(thr),
            "Precision_1": m["Precision_1"],
            "Recall_1": m["Recall_1"],
            "Fbeta": fbeta,
            "FPR": m["FPR_fake_violent"],
            "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"]
        }

        if row["Precision_1"] >= min_precision:
            if (best is None) or (row["Fbeta"] > best["Fbeta"]) or (
                row["Fbeta"] == best["Fbeta"] and row["FPR"] < best["FPR"]
            ):
                best = row

        if (best_any is None) or (row["Precision_1"] > best_any["Precision_1"]) or (
            row["Precision_1"] == best_any["Precision_1"] and row["Fbeta"] > best_any["Fbeta"]
        ):
            best_any = row

    chosen = best if best is not None else best_any
    chosen["meets_min_precision"] = (best is not None)
    return chosen


# ==============================================================================
# MANUAL THEORY (ALIGNED): DISCRETE IG + DISCRETE NB
# (EDUCATIONAL DEMO on TRAIN only; not the "final model")
# ==============================================================================
def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def information_gain_discrete(Xd: pd.DataFrame, y: pd.Series, feature: str) -> Tuple[float, float, float]:
    y_counts = y.value_counts().sort_index().values
    H_y = entropy_from_counts(y_counts)

    weighted = 0.0
    for _, idx in Xd.groupby(feature).groups.items():
        sub_y = y.loc[idx]
        sub_counts = sub_y.value_counts().sort_index().values
        weighted += (len(sub_y) / len(y)) * entropy_from_counts(sub_counts)

    gain = H_y - weighted
    return gain, H_y, weighted


def discretize_for_manual_demo(X: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    Xd = X.copy()

    if "hour" in Xd.columns:
        Xd["hour"] = pd.cut(
            Xd["hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"]
        )

    if "Vict Age" in Xd.columns:
        Xd["Vict Age"] = pd.cut(
            Xd["Vict Age"],
            bins=[-1, 17, 29, 44, 59, 200],
            labels=["0-17", "18-29", "30-44", "45-59", "60+"]
        )

    if "AREA" in Xd.columns:
        try:
            Xd["AREA"] = Xd["AREA"].astype("Int64").astype(str)
        except Exception:
            Xd["AREA"] = Xd["AREA"].astype(str)

    demo_cols = [c for c in cfg.IG_FEATURES_TO_ANALYZE if c in Xd.columns]
    for c in demo_cols:
        Xd[c] = Xd[c].astype(str)

    return Xd


def manual_discrete_naive_bayes_demo(
    Xd: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    sample_row: pd.Series,
    laplace: float = 1.0
) -> Dict[str, Any]:
    classes = sorted(y.unique())
    priors = {c: float((y == c).mean()) for c in classes}
    tables = {f: pd.crosstab(y, Xd[f]) for f in features}

    log_post = {}
    for c in classes:
        logp = math.log(priors[c] + 1e-12)
        for f in features:
            val = str(sample_row[f])
            ct = tables[f]
            num_values = ct.shape[1]
            count_cv = ct.loc[c, val] if (c in ct.index and val in ct.columns) else 0
            count_c = ct.loc[c].sum() if c in ct.index else 0
            prob = (count_cv + laplace) / (count_c + laplace * num_values + 1e-12)
            logp += math.log(prob + 1e-12)
        log_post[c] = logp

    maxlog = max(log_post.values())
    exps = {c: math.exp(v - maxlog) for c, v in log_post.items()}
    Z = sum(exps.values())
    post = {c: exps[c] / (Z + 1e-12) for c in classes}
    pred = max(post, key=post.get)

    return {"priors": priors, "posteriors": post, "prediction": pred}


# ==============================================================================
# PEBLS (FAST): VDM + tuning + candidate prefilter + vectorized distance
# ==============================================================================
class PEBLSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        r: int = 1,
        p: int = 2,
        k: int = 1,
        epochs: int = 2,
        max_exemplars: int = 25_000,
        tune_exemplars: int = 3_000,
        tune_candidates: int = 2_000,
        pred_max_candidates: int = 5_000,
        pred_fallback_candidates: int = 5_000,
        random_state: int = 42
    ):
        self.r = r
        self.p = p
        self.k = k
        self.epochs = epochs
        self.max_exemplars = max_exemplars
        self.tune_exemplars = tune_exemplars
        self.tune_candidates = tune_candidates
        self.pred_max_candidates = pred_max_candidates
        self.pred_fallback_candidates = pred_fallback_candidates
        self.random_state = random_state

        self.features_: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.class_to_idx_: Dict[Any, int] = {}

        self.val_to_code_: List[Dict[str, int]] = []
        self.code_to_val_: List[List[str]] = []
        self.dist_mats_: List[np.ndarray] = []
        self.inv_index_: List[Dict[int, np.ndarray]] = []

        self.X_code_: Optional[np.ndarray] = None
        self.y_ex_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.uses_: Optional[np.ndarray] = None
        self.correct_: Optional[np.ndarray] = None
        self.tune_idx_: Optional[np.ndarray] = None

    def _build_value_maps(self, Xs: pd.DataFrame) -> None:
        self.val_to_code_ = []
        self.code_to_val_ = []
        for f in self.features_:
            vals = pd.unique(Xs[f].astype(str).fillna("__UNK__"))
            vals = [str(v) for v in vals]
            if "__UNK__" not in vals:
                vals.append("__UNK__")
            vals_sorted = sorted(vals)
            v2c = {v: i for i, v in enumerate(vals_sorted)}
            self.val_to_code_.append(v2c)
            self.code_to_val_.append(vals_sorted)

    def _compute_vdm_dist_mats(self, Xs: pd.DataFrame, y: pd.Series) -> None:
        self.classes_ = np.array(sorted(y.unique()))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        K = len(self.classes_)
        self.dist_mats_ = []

        for fi, f in enumerate(self.features_):
            vals = self.code_to_val_[fi]
            V = len(vals)
            probs = np.zeros((V, K), dtype=np.float64)

            for vi, v in enumerate(vals):
                if v == "__UNK__":
                    continue
                mask = (Xs[f].astype(str) == v)
                sub_y = y[mask]
                vec = np.zeros(K, dtype=np.float64)
                vc = sub_y.value_counts()
                for c, cnt in vc.items():
                    vec[self.class_to_idx_[c]] = cnt
                total = vec.sum()
                probs[vi] = (vec + 1.0) / (total + K)

            probs[self.val_to_code_[fi]["__UNK__"]] = np.ones(K) / max(K, 1)

            dist = np.zeros((V, V), dtype=np.float64)
            for i in range(V):
                diff = np.abs(probs[i] - probs)
                if self.r == 1:
                    dist[i] = diff.sum(axis=1)
                else:
                    dist[i] = (diff ** self.r).sum(axis=1)

            self.dist_mats_.append(dist)

    def _encode_X(self, Xs: pd.DataFrame) -> np.ndarray:
        X_code = np.zeros((len(Xs), len(self.features_)), dtype=np.int32)
        for fi, f in enumerate(self.features_):
            v2c = self.val_to_code_[fi]
            unk = v2c.get("__UNK__", 0)
            col = Xs[f].astype(str).fillna("__UNK__").values
            X_code[:, fi] = np.array([v2c.get(str(v), unk) for v in col], dtype=np.int32)
        return X_code

    def _build_inverted_index(self, X_code: np.ndarray) -> None:
        self.inv_index_ = []
        n, m = X_code.shape
        for fi in range(m):
            mapping: Dict[int, List[int]] = {}
            codes = X_code[:, fi]
            for idx in range(n):
                c = int(codes[idx])
                mapping.setdefault(c, []).append(idx)
            self.inv_index_.append({k: np.array(v, dtype=np.int32) for k, v in mapping.items()})

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PEBLSClassifier expects X as pandas DataFrame.")
        self.features_ = list(X.columns)

        rng = np.random.default_rng(self.random_state)

        if len(X) > self.max_exemplars:
            idx_keep = []
            yv = y.values
            for c in sorted(np.unique(yv)):
                idx_c = np.where(yv == c)[0]
                take = max(1, int(self.max_exemplars * (len(idx_c) / len(X))))
                take = min(take, len(idx_c))
                idx_keep.append(rng.choice(idx_c, size=take, replace=False))
            idx_keep = np.concatenate(idx_keep)
            rng.shuffle(idx_keep)
            X = X.iloc[idx_keep].copy()
            y = y.iloc[idx_keep].copy()

        Xs = X.copy()
        for f in self.features_:
            Xs[f] = Xs[f].astype(str).fillna("__UNK__")

        self._build_value_maps(Xs)
        self._compute_vdm_dist_mats(Xs, y)

        self.X_code_ = self._encode_X(Xs)
        self.y_ex_ = y.values

        n = len(Xs)
        self.uses_ = np.ones(n, dtype=np.float64)
        self.correct_ = np.ones(n, dtype=np.float64)
        self.weights_ = self.uses_ / self.correct_

        self._build_inverted_index(self.X_code_)

        tune_n = min(self.tune_exemplars, n)
        self.tune_idx_ = rng.choice(np.arange(n), size=tune_n, replace=False)
        self._tune_weights_fast()

        return self

    def _tune_weights_fast(self) -> None:
        if self.epochs <= 0 or self.tune_idx_ is None:
            return

        rng = np.random.default_rng(self.random_state)
        n = len(self.y_ex_)
        all_idx = np.arange(n)
        cand_size = min(self.tune_candidates, n)

        for _ in range(self.epochs):
            order = self.tune_idx_.copy()
            rng.shuffle(order)

            for i in order:
                x_i = self.X_code_[i]
                y_true = self.y_ex_[i]

                cand = rng.choice(all_idx, size=cand_size, replace=False)
                cand = cand[cand != i]
                if len(cand) == 0:
                    continue

                d = self._distance_candidates(x_i, cand)
                best_j = cand[int(np.argmin(d))]
                y_pred = self.y_ex_[best_j]

                self.uses_[best_j] += 1.0
                if y_pred == y_true:
                    self.correct_[best_j] += 1.0
                self.weights_[best_j] = self.uses_[best_j] / max(self.correct_[best_j], 1.0)

    def _distance_candidates(self, x_code: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
        s = np.zeros(len(cand_idx), dtype=np.float64)
        for fi in range(len(self.features_)):
            cx = int(x_code[fi])
            cy = self.X_code_[cand_idx, fi]
            d = self.dist_mats_[fi][cx, cy]
            s += (d ** self.p)
        s *= self.weights_[cand_idx]
        return s

    def _candidates_for_x(self, x_code: np.ndarray) -> np.ndarray:
        hits = []
        for fi in range(len(self.features_)):
            code = int(x_code[fi])
            arr = self.inv_index_[fi].get(code)
            if arr is not None:
                hits.append(arr)
        if not hits:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate(hits)).astype(np.int32)

    def _predict_one(self, x_code: np.ndarray) -> Any:
        rng = np.random.default_rng(self.random_state)
        cand = self._candidates_for_x(x_code)

        if cand.size == 0:
            n = len(self.y_ex_)
            take = min(self.pred_fallback_candidates, n)
            cand = rng.choice(np.arange(n), size=take, replace=False).astype(np.int32)

        if cand.size > self.pred_max_candidates:
            cand = rng.choice(cand, size=self.pred_max_candidates, replace=False).astype(np.int32)

        d = self._distance_candidates(x_code, cand)

        k = max(1, int(self.k))
        k = min(k, len(cand))
        nn_pos = np.argpartition(d, kth=k - 1)[:k]
        nn_idx = cand[nn_pos]
        nn_d = d[nn_pos]

        votes: Dict[Any, float] = {}
        for j, dist in zip(nn_idx, nn_d):
            cls = self.y_ex_[j]
            w = 1.0 / (dist + 1e-12)
            votes[cls] = votes.get(cls, 0.0) + w

        return max(votes, key=votes.get)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.X_code_ is None:
            raise RuntimeError("PEBLSClassifier is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PEBLSClassifier expects X as pandas DataFrame.")

        Xs = X.copy()
        for f in self.features_:
            Xs[f] = Xs[f].astype(str).fillna("__UNK__")

        X_code = self._encode_X(Xs)
        preds = [self._predict_one(X_code[i]) for i in range(len(X_code))]
        return np.array(preds)


# ==============================================================================
# PIPELINE BUILDERS
# ==============================================================================
def make_onehot_encoder(max_categories: int = 10) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=max_categories,
            sparse_output=True
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def infer_num_cat_cols(X_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    return num_cols, cat_cols


def build_dt_model(cfg: Config, X_small: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols = infer_num_cat_cols(X_small)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="mean"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_onehot_encoder(cfg.MAX_CATEGORIES_ONEHOT))
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=cfg.DT_MAX_DEPTH,
        min_samples_split=cfg.DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=cfg.DT_MIN_SAMPLES_LEAF,
        random_state=cfg.RANDOM_STATE
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_nb_model(cfg: Config, X_small: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols = infer_num_cat_cols(X_small)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="mean"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    def to_dense(x):
        return np.asarray(x)

    return Pipeline([("pre", pre), ("to_dense", TransformerMixinWrapper(to_dense)), ("clf", GaussianNB())])


def build_knn_model(cfg: Config, X_small: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols = infer_num_cat_cols(X_small)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler(with_mean=True))
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_onehot_encoder(cfg.MAX_CATEGORIES_ONEHOT))
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return Pipeline([("pre", pre), ("clf", KNeighborsClassifier(n_neighbors=cfg.KNN_K, n_jobs=-1))])


def build_pebls_model(cfg: Config) -> Pipeline:
    return Pipeline([
        ("select", SelectColumns(cfg.PEBLS_FEATURES)),
        ("imp", ImputeMostFrequentDF()),
        ("cast", CastToStrDF()),
        ("clf", PEBLSClassifier(
            r=1, p=2, k=1,
            epochs=cfg.PEBLS_EPOCHS,
            max_exemplars=cfg.PEBLS_MAX_EXEMPLARS,
            tune_exemplars=cfg.PEBLS_TUNE_EXEMPLARS,
            tune_candidates=cfg.PEBLS_TUNE_CANDIDATES,
            pred_max_candidates=cfg.PEBLS_PRED_MAX_CANDIDATES,
            pred_fallback_candidates=cfg.PEBLS_PRED_FALLBACK_CANDIDATES,
            random_state=cfg.RANDOM_STATE
        ))
    ])


# ==============================================================================
# MODEL META
# ==============================================================================
def model_meta_info(model_name: str, fitted_model_pipeline: Pipeline) -> Dict[str, str]:
    meta = {"Comprehensibility": "", "Complexity": ""}

    if "Decision Tree" in model_name:
        clf: DecisionTreeClassifier = fitted_model_pipeline.named_steps["clf"]
        meta["Comprehensibility"] = f"High: depth={clf.get_depth()}, leaves={clf.get_n_leaves()}"
        meta["Complexity"] = "Train ~ O(n·m·log n), Predict ~ O(depth)"
    elif "Naive Bayes" in model_name:
        meta["Comprehensibility"] = "High: probabilistic; independence assumption"
        meta["Complexity"] = "Train ~ O(n·m), Predict ~ O(m)"
    elif "KNN" in model_name:
        k = fitted_model_pipeline.named_steps["clf"].n_neighbors
        meta["Comprehensibility"] = f"Medium: instance-based; k={k}"
        meta["Complexity"] = "Train ~ O(1), Predict ~ O(n·m)"
    elif "PEBLS" in model_name:
        clf: PEBLSClassifier = fitted_model_pipeline.named_steps["clf"]
        meta["Comprehensibility"] = (
            f"Medium-High: nearest exemplar + VDM; exemplars={len(clf.y_ex_)}, "
            f"cand_max={clf.pred_max_candidates}, epochs={clf.epochs}"
        )
        meta["Complexity"] = "Train ~ O(n·m + epochs·tune·cand·m), Predict ~ O(cand·m)"
    else:
        meta["Comprehensibility"] = "N/A"
        meta["Complexity"] = "N/A"

    return meta


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    cfg = Config()
    ensure_dir(cfg.RESULTS_DIR)

    print_section("STEP 1: LOAD DATA")
    df = pd.read_csv(cfg.DATA_PATH)

    if len(df) > cfg.SAMPLE_SIZE:
        df = df.sample(n=cfg.SAMPLE_SIZE, random_state=cfg.RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled to {len(df):,} rows")

    if cfg.TARGET_COLUMN not in df.columns:
        raise ValueError(f"TARGET_COLUMN '{cfg.TARGET_COLUMN}' not found.")

    y = df[cfg.TARGET_COLUMN].astype(int)
    X_raw = df.drop(columns=[cfg.TARGET_COLUMN])

    print(f"X: {X_raw.shape} | y: {y.shape}")
    print(f"Violent=1: {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

    print_section("STEP 2: SPLIT (TRAIN/VAL/TEST)")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    val_size_adjusted = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted,
        random_state=cfg.RANDOM_STATE, stratify=y_trainval
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    print_section("STEP 3: BUILD PIPELINES (NO LEAKAGE + HARD LEAKAGE DROP)")

    # Combine drops (high-cardinality + explicit leakage)
    drop_cols = tuple(dict.fromkeys(list(cfg.HIGH_CARDINALITY_COLS) + list(cfg.LEAKAGE_COLS)))

    # sanity log: show if any leakage columns exist in your X
    present_leak = [c for c in cfg.LEAKAGE_COLS if c in X_train.columns]
    if present_leak:
        print("LEAKAGE COLS PRESENT in raw X -> will be dropped:", present_leak)
    else:
        print("LEAKAGE COLS not present in raw X (still enforced in pipeline).")

    X_small = DropColumns(drop_cols).transform(X_train.head(2000).copy())

    dt_pipe = Pipeline([("drop", DropColumns(drop_cols)), ("model", build_dt_model(cfg, X_small))])
    nb_pipe = Pipeline([("drop", DropColumns(drop_cols)), ("model", build_nb_model(cfg, X_small))])
    knn_pipe = Pipeline([("drop", DropColumns(drop_cols)), ("model", build_knn_model(cfg, X_small))])
    pebls_pipe = Pipeline([("drop", DropColumns(drop_cols)), ("model", build_pebls_model(cfg))])

    models: Dict[str, Pipeline] = {
        "Naive Bayes (Gaussian)": nb_pipe,
        "KNN": knn_pipe,
        "PEBLS (VDM + tuning)": pebls_pipe,
        "Decision Tree (IG)": dt_pipe,
    }

    for name in models:
        print(f" - {name}")

    print_section("STEP 4: MANUAL THEORY (ALIGNED) - TRAIN ONLY (EDU DEMO)")

    # Important: manual demo uses TRAIN only (no leakage) and uses dropped features
    X_train_demo = DropColumns(drop_cols).transform(X_train.copy())
    Xd = discretize_for_manual_demo(X_train_demo, cfg)
    demo_features = [c for c in cfg.IG_FEATURES_TO_ANALYZE if c in Xd.columns]

    if demo_features:
        ig_rows = []
        for f in demo_features:
            gain, Hy, HyA = information_gain_discrete(
                Xd[demo_features].reset_index(drop=True),
                y_train.reset_index(drop=True),
                f
            )
            ig_rows.append({"Feature": f, "H(Y)": Hy, "H(Y|A)": HyA, "IG": gain})

        ig_df = pd.DataFrame(ig_rows).sort_values("IG", ascending=False)
        print("\nInformation Gain (DISCRETE):")
        print(ig_df.to_string(index=False))

        # NB demo: still ok to show on a sample row, but training tables are from TRAIN
        X_test_demo = discretize_for_manual_demo(DropColumns(drop_cols).transform(X_test.copy()), cfg)
        sample_idx = X_test_demo.index[0]
        sample_row = X_test_demo.loc[sample_idx, demo_features]

        nb_demo = manual_discrete_naive_bayes_demo(
            Xd[demo_features].reset_index(drop=True),
            y_train.reset_index(drop=True),
            demo_features,
            sample_row
        )
        print("\nDiscrete Naive Bayes Demo:")
        print(f"Features: {demo_features}")
        print(f"Priors: {nb_demo['priors']}")
        print(f"Posteriors: {nb_demo['posteriors']}")
        print(f"Prediction: {nb_demo['prediction']} | Actual: {int(y_test.loc[sample_idx])}")
    else:
        print("Manual demo skipped: demo features not found after dropping columns.")

    print_section("STEP 5: DECISION TREE PRUNING (ccp_alpha via GridSearchCV)")

    scorer_fbeta = make_scorer(fbeta_score, beta=cfg.FBETA, pos_label=1, zero_division=0)

    alpha_grid = np.geomspace(1e-6, 1e-2, num=cfg.PRUNING_GRID_SIZE)
    grid = GridSearchCV(
        dt_pipe,
        param_grid={"model__clf__ccp_alpha": alpha_grid},
        scoring=scorer_fbeta,
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_dt = grid.best_estimator_
    best_alpha = grid.best_params_["model__clf__ccp_alpha"]
    print(f"Best alpha: {best_alpha:.8f} (scored by F{cfg.FBETA} on class=1)")

    models["Decision Tree (IG, Pruned)"] = best_dt
    models.pop("Decision Tree (IG)", None)

    print_section("STEP 6: FIT + VAL THRESHOLD TUNING (NO 'FAKE' VIOLENT) + TEST EVAL")

    label_names = ["Non-Violent (0)", "Violent (1)"]
    results = []

    for name, pipe in models.items():
        print("\n" + "-" * 78)
        print(name)

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        thr_info = None
        if hasattr(pipe, "predict_proba"):
            proba_val = pipe.predict_proba(X_val)[:, 1]
            thr_info = tune_threshold_for_precision(
                y_true=y_val.values,
                proba_1=proba_val,
                thresholds=list(cfg.THRESHOLDS),
                beta=cfg.FBETA,
                min_precision=cfg.MIN_PRECISION_VIOLENT_VAL
            )
            status = "OK" if thr_info["meets_min_precision"] else "NO (fallback: max precision)"
            print(
                f"VAL threshold tuned -> thr={thr_info['threshold']:.2f} | "
                f"P1={thr_info['Precision_1']:.3f} R1={thr_info['Recall_1']:.3f} "
                f"F{cfg.FBETA}={thr_info['Fbeta']:.3f} | meets_min_precision={status}"
            )
        else:
            print("VAL threshold tuning skipped (no predict_proba).")

        t1 = time.time()
        if thr_info is not None:
            proba_test = pipe.predict_proba(X_test)[:, 1]
            y_pred = (proba_test >= thr_info["threshold"]).astype(int)
        else:
            y_pred = pipe.predict(X_test)
        pred_time = time.time() - t1

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        m = binary_metrics_from_cm(cm)
        fbeta_test = fbeta_from_pr(m["Precision_1"], m["Recall_1"], cfg.FBETA)

        prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        cm_path = f"{cfg.RESULTS_DIR}/cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_confusion_matrix(cm, label_names, f"Confusion Matrix - {name}", cm_path, dpi=cfg.FIGURE_DPI)

        meta = model_meta_info(name, pipe.named_steps["model"])

        print(f"TEST Accuracy={m['Accuracy']:.4f} | BalancedAcc={m['Balanced_Acc']:.4f}")
        print(
            f"TEST Violent(1): Precision={m['Precision_1']:.4f} | Recall={m['Recall_1']:.4f} | "
            f"F1={m['F1_1']:.4f} | F{cfg.FBETA}={fbeta_test:.4f}"
        )
        print(f"TEST Fake-violent rate (FPR)={m['FPR_fake_violent']:.4f}  (lower = better)")
        print(f"Weighted: P_w={prec_w:.4f} R_w={rec_w:.4f} F1_w={f1_w:.4f} | Macro-F1={f1_macro:.4f}")
        print(f"Train={train_time:.2f}s | Predict={pred_time:.2f}s")
        print(f"Saved CM: {cm_path}")
        print(f"Comprehensibility: {meta['Comprehensibility']}")
        print(f"Complexity: {meta['Complexity']}")

        results.append({
            "Model": name,
            "Threshold_VAL": (thr_info["threshold"] if thr_info is not None else np.nan),
            "Meets_MinPrec_VAL": (thr_info["meets_min_precision"] if thr_info is not None else np.nan),

            "Accuracy": m["Accuracy"],
            "Balanced_Acc": m["Balanced_Acc"],
            "Precision_1": m["Precision_1"],
            "Recall_1": m["Recall_1"],
            "F1_1": m["F1_1"],
            f"F{cfg.FBETA}_1": fbeta_test,
            "FPR_fake_violent": m["FPR_fake_violent"],

            "Precision_w": prec_w,
            "Recall_w": rec_w,
            "F1_w": f1_w,
            "F1_macro": f1_macro,

            "Train_Time_s": train_time,
            "Predict_Time_s": pred_time,
            "Comprehensibility": meta["Comprehensibility"],
            "Complexity": meta["Complexity"]
        })

        gc.collect()

    comparison_df = pd.DataFrame(results).sort_values(
        by=[f"F{cfg.FBETA}_1", "FPR_fake_violent"],
        ascending=[False, True]
    )

    out_csv = f"{cfg.RESULTS_DIR}/model_comparison_practice.csv"
    comparison_df.to_csv(out_csv, index=False)

    print_section("STEP 7: MODEL COMPARISON 'IN PRACTICE' (PRIORITIZE: DON'T FAKE VIOLENT)")
    print(comparison_df.to_string(index=False))
    print(f"Saved: {out_csv}")

    plt.figure(figsize=(12, 5))
    x = np.arange(len(comparison_df))
    plt.bar(x - 0.25, comparison_df["Precision_1"].values, width=0.25, label="Precision_1")
    plt.bar(x, comparison_df["Recall_1"].values, width=0.25, label="Recall_1")
    plt.bar(x + 0.25, (1.0 - comparison_df["FPR_fake_violent"].values), width=0.25, label="(1 - FPR_fake_violent)")
    plt.xticks(x, comparison_df["Model"].values, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Practical: Precision(violent), Recall(violent), and (1-FPR) (higher better)")
    plt.legend()
    plt.tight_layout()
    plot_path = f"{cfg.RESULTS_DIR}/model_comparison_practice.png"
    plt.savefig(plot_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    print_section("STEP 8: CROSS-VALIDATION (SUBSET) - F0.5 on class=1 (precision-emphasis)")
    X_cv = X_train.sample(n=min(cfg.CV_SAMPLE_SIZE, len(X_train)), random_state=cfg.RANDOM_STATE)
    y_cv = y_train.loc[X_cv.index]
    skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    for name, pipe in models.items():
        t0 = time.time()
        scores = cross_val_score(pipe, X_cv, y_cv, cv=skf, scoring=scorer_fbeta, n_jobs=-1)
        dt = time.time() - t0
        print(f"{name}: mean F{cfg.FBETA}={scores.mean():.4f}, std={scores.std():.4f}, time={dt:.1f}s")

    print_section("DONE")
    print("Generated in ./results:")
    print(" - cm_*.png")
    print(" - model_comparison_practice.csv")
    print(" - model_comparison_practice.png")
    print("\nNOTE:")
    print(f" - Leakage cols are force-dropped: {list(cfg.LEAKAGE_COLS)}")
    print(f" - To reduce 'fake violent' even more, increase MIN_PRECISION_VIOLENT_VAL (current={cfg.MIN_PRECISION_VIOLENT_VAL}).")
    print(" - Threshold tuning is only on VAL (no leakage). Final numbers are TEST only.")


if __name__ == "__main__":
    main()