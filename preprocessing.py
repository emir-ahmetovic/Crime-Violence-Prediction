import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --------------------------------------------
# Configuration
# --------------------------------------------
INPUT_FILE = "crime_full.csv"

# EDA output keeps all columns (for plots/inspection)
OUTPUT_EDA_FILE = "clean_eda.csv"

# MODEL output is leakage-safe for classification
OUTPUT_MODEL_FILE = "clean.csv"

EDA_DIR = "eda_outputs"
os.makedirs(EDA_DIR, exist_ok=True)

# --------------------------------------------
# 1. Load Data
# --------------------------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# --------------------------------------------
# 2. Dataset Overview & Data Quality Checks
# --------------------------------------------
print("\n=== DATASET OVERVIEW ===")
print(df.info(memory_usage="deep"))

print("\n=== MISSING VALUES (%) ===")
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print(missing_pct)

plt.figure(figsize=(14, 6))
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/missing_values_heatmap.png")
plt.close()

# --------------------------------------------
# 3. Data Type Fixes
# --------------------------------------------
print("\nFixing data types...")

# Dates
df["Date Rptd"] = pd.to_datetime(df.get("Date Rptd"), errors="coerce")
df["DATE OCC"] = pd.to_datetime(df.get("DATE OCC"), errors="coerce")

# TIME OCC -> hour
df["TIME OCC"] = pd.to_numeric(df.get("TIME OCC"), errors="coerce")
df["hour"] = (df["TIME OCC"] // 100)
df.loc[(df["hour"] < 0) | (df["hour"] > 23), "hour"] = np.nan  # keep clean
df["hour"] = df["hour"].astype("Int64")

# Numeric columns (convert if exist)
numeric_cols = [
    "Vict Age", "LAT", "LON", "AREA", "Rpt Dist No",
    "Crm Cd", "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4",
    "Premis Cd", "Weapon Used Cd", "Part 1-2"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------
# 4. Basic Cleaning (Deterministic, no global-stat imputation)
# --------------------------------------------
print("\nBasic cleaning...")

# Categorical -> UNKNOWN (deterministic)
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

# Vict Age: keep NaN for missing/invalid (handle later in Pipeline)
if "Vict Age" in df.columns:
    df.loc[(df["Vict Age"] <= 0) | (df["Vict Age"] > 120), "Vict Age"] = np.nan

# Vict Sex: no random replacement (avoid injecting noise)
if "Vict Sex" in df.columns:
    df["Vict Sex"] = df["Vict Sex"].astype(str).str.upper().str.strip()
    df.loc[df["Vict Sex"].isin(["X", "H"]), "Vict Sex"] = "UNKNOWN"
    df.loc[~df["Vict Sex"].isin(["F", "M", "UNKNOWN"]), "Vict Sex"] = "UNKNOWN"

# --------------------------------------------
# 5. Feature Engineering (Allowed)
# --------------------------------------------
print("\nFeature engineering...")

# Date-derived features (ok; then original date fields will be dropped from MODEL output)
for prefix, col in [("rpt", "Date Rptd"), ("occ", "DATE OCC")]:
    if col in df.columns:
        df[f"{prefix}_year"] = df[col].dt.year.astype("Int64")
        df[f"{prefix}_month"] = df[col].dt.month.astype("Int64")
        df[f"{prefix}_day"] = df[col].dt.day.astype("Int64")
        df[f"{prefix}_day_of_week"] = df[col].dt.dayofweek.astype("Int64")
        df[f"{prefix}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype("Int64")

# Season from occ_month
def get_season(month):
    if pd.isna(month):
        return "UNKNOWN"
    month = int(month)
    if month in [12, 1, 2]:
        return "WINTER"
    if month in [3, 4, 5]:
        return "SPRING"
    if month in [6, 7, 8]:
        return "SUMMER"
    return "FALL"

if "occ_month" in df.columns:
    df["season"] = df["occ_month"].apply(get_season)
else:
    df["season"] = "UNKNOWN"

# Reporting delay (keep NaN if dates missing)
if "Date Rptd" in df.columns and "DATE OCC" in df.columns:
    df["reporting_delay"] = (df["Date Rptd"] - df["DATE OCC"]).dt.days
    # negative delays -> 0, but keep NaN if missing
    df.loc[df["reporting_delay"] < 0, "reporting_delay"] = 0
else:
    df["reporting_delay"] = np.nan

# Time period (categorical)
def time_period_from_hour(h):
    if pd.isna(h):
        return "UNKNOWN"
    h = int(h)
    if h < 6:
        return "NIGHT"
    if h < 12:
        return "MORNING"
    if h < 18:
        return "AFTERNOON"
    return "EVENING"

df["time_period"] = df["hour"].apply(time_period_from_hour)

# OPTIONAL: Keep these for EDA only (they are proxies; will be dropped from MODEL output)
if "Weapon Used Cd" in df.columns:
    df["is_armed"] = df["Weapon Used Cd"].notna().astype(int)
else:
    df["is_armed"] = 0

if "Part 1-2" in df.columns:
    df["crime_severity"] = df["Part 1-2"].map({1: "SERIOUS", 2: "LESS_SERIOUS"}).fillna("UNKNOWN")
else:
    df["crime_severity"] = "UNKNOWN"

# --------------------------------------------
# 6. Target Variable
# --------------------------------------------
print("\nCreating target variable...")

violent_codes = {
    110, 111, 112, 113, 121, 122,
    230, 231, 235, 236,
    250, 251, 761, 926
}

df["is_violent_crime"] = df["Crm Cd"].isin(violent_codes).astype(int)
print(df["is_violent_crime"].value_counts())

# --------------------------------------------
# 7. EDA Plots (use full df)
# --------------------------------------------
print("\nGenerating EDA plots...")

plt.figure(figsize=(6, 4))
sns.countplot(x="is_violent_crime", data=df)
plt.title("Violent vs Non-Violent Crimes")
plt.savefig(f"{EDA_DIR}/target_distribution.png")
plt.close()

df.groupby("hour").size().plot(kind="line", figsize=(8, 4))
plt.title("Crimes by Hour")
plt.savefig(f"{EDA_DIR}/crimes_by_hour.png")
plt.close()

if "occ_day_of_week" in df.columns:
    df.groupby("occ_day_of_week").size().plot(kind="bar")
    plt.title("Crimes by Day of Week")
    plt.savefig(f"{EDA_DIR}/crimes_by_day.png")
    plt.close()

if "occ_month" in df.columns:
    df.groupby("occ_month").size().plot(kind="bar")
    plt.title("Crimes by Month")
    plt.savefig(f"{EDA_DIR}/crimes_by_month.png")
    plt.close()

if "Crm Cd Desc" in df.columns:
    df["Crm Cd Desc"].value_counts().head(10).plot(kind="barh")
    plt.title("Top 10 Crime Types")
    plt.savefig(f"{EDA_DIR}/top_crime_types.png")
    plt.close()

if "AREA NAME" in df.columns:
    df["AREA NAME"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Areas by Crime")
    plt.savefig(f"{EDA_DIR}/top_areas.png")
    plt.close()

if "Vict Age" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Vict Age"], bins=40)
    plt.title("Victim Age Distribution")
    plt.savefig(f"{EDA_DIR}/victim_age_distribution.png")
    plt.close()

if "Vict Sex" in df.columns:
    df["Vict Sex"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Victim Sex Distribution")
    plt.ylabel("")
    plt.savefig(f"{EDA_DIR}/victim_sex_distribution.png")
    plt.close()

if "Weapon Desc" in df.columns:
    df["Weapon Desc"].value_counts().head(10).plot(kind="bar")
    plt.title("Top Weapons Used")
    plt.savefig(f"{EDA_DIR}/weapon_usage.png")
    plt.close()

if "Premis Desc" in df.columns:
    df["Premis Desc"].value_counts().head(10).plot(kind="bar")
    plt.title("Top Premise Types")
    plt.savefig(f"{EDA_DIR}/premise_types.png")
    plt.close()

# Geographic scatter (drop NaNs for plotting only)
if "LON" in df.columns and "LAT" in df.columns:
    tmp = df.dropna(subset=["LON", "LAT"]).sample(
        min(20000, df.dropna(subset=["LON", "LAT"]).shape[0]),
        random_state=42
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="LON", y="LAT", hue="is_violent_crime", data=tmp, alpha=0.5, legend=False)
    plt.title("Crime Locations (Sampled)")
    plt.savefig(f"{EDA_DIR}/crime_map.png")
    plt.close()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["int64", "float64", "Int64"])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Numeric Feature Correlations")
plt.savefig(f"{EDA_DIR}/correlation_heatmap.png")
plt.close()

# --------------------------------------------
# 8. SAVE OUTPUTS
# --------------------------------------------
print("\nSaving outputs...")

# Save full EDA dataset
df.to_csv(OUTPUT_EDA_FILE, index=False)

# Build leakage-safe MODEL dataset
LEAKY_PROXY_COLS = [
    # label definition sources (MUST DROP from X)
    "Crm Cd", "Crm Cd Desc", "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4",
    "Part 1-2",
    "Weapon Used Cd", "Weapon Desc",
    "is_armed",
    "crime_severity",

    # IDs / high-card / text fields (recommended to drop)
    "DR_NO", "Mocodes", "LOCATION", "Cross Street",
    "AREA NAME", "Premis Desc", "Status Desc",

    # original date/time strings (because we keep engineered versions)
    "Date Rptd", "DATE OCC", "TIME OCC"
]

df_model = df.copy()
df_model = df_model.drop(columns=[c for c in LEAKY_PROXY_COLS if c in df_model.columns], errors="ignore")

# ensure target present
if "is_violent_crime" not in df_model.columns:
    raise RuntimeError("Target is missing in df_model.")

# Save model-ready dataset (this is what you use for classification)
df_model.to_csv(OUTPUT_MODEL_FILE, index=False)

print("\n===================================")
print("PREPROCESSING & EDA COMPLETE")
print("Saved EDA file:   ", OUTPUT_EDA_FILE)
print("Saved MODEL file: ", OUTPUT_MODEL_FILE, " (USE THIS FOR CLASSIFICATION)")
print("EDA outputs in:   ", EDA_DIR)
print("MODEL columns:", df_model.shape[1], "| rows:", df_model.shape[0])
print("===================================")
