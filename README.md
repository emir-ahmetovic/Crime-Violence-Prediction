# Crime Data Classification Project

A machine learning project that analyzes Los Angeles crime data (2020-present) and builds predictive models to classify crimes as violent or non-violent.

## Project Overview

This project uses multiple classification algorithms to predict whether a crime incident is violent based on features such as:
- **Temporal patterns**: Time of day, day of week, season
- **Victim demographics**: Age, gender
- **Location data**: Crime area/district
- **Weapon information**: Whether a weapon was used
- **Premise type**: Location category (e.g., street, residence, vehicle)
- **Engineered features**: Time-of-day risk groups, victim age risk groups, area-time interactions

## Dataset

- **Source**: LA Crime Data from 2020 to Present
- **Files**: 
  - `crime_full.csv` - Raw data
  - `clean.csv` - Preprocessed data for model training (leakage-safe)
  - `clean_eda.csv` - Preprocessed data for exploratory analysis

## Project Structure

```
├── preprocessing.py          # Data cleaning, feature engineering, and preparation
├── eda.py                   # Exploratory data analysis and visualizations
├── classification.py        # ML model training, tuning, and evaluation
├── eda_outputs/             # Generated visualizations and analysis plots
└── results/                 # Model evaluation results and metrics
```

## Key Components

### 1. Data Preprocessing (`preprocessing.py`)
- Loads raw crime data
- Handles missing values
- Creates engineered features:
  - Temporal features (hour, day of week, time periods)
  - Victim risk groups (age-based segments)
  - Area-time interactions
  - Weapon indicators
- Generates clean datasets with and without data leakage checks
- Produces visualizations for data quality assessment

### 2. Exploratory Data Analysis (`eda.py`)
- Analyzes target distribution (violent vs. non-violent crimes)
- Crime patterns by:
  - Time of day
  - Geographic area
  - Victim age
  - Weapon usage
- Generates matplotlib/seaborn visualizations saved to `eda_outputs/`

### 3. Classification Models (`classification.py`)
Trains and evaluates multiple algorithms:

- **Decision Tree** with hyperparameter tuning
  - Max depth: [8, 10, 12, 15]
  - Min samples split: [100, 200, 300]
  - Cost-complexity pruning for generalization
  
- **Random Forest** - Ensemble of decision trees
  - 100 estimators
  - Max depth: 15
  
- **Gradient Boosting** - Sequential tree ensemble
  - 100 estimators
  - Learning rate: 0.1
  
- **Naive Bayes** - Probabilistic classifier (parameter-free)

- **K-Nearest Neighbors** - Instance-based learning
  - k = 7 neighbors
  
- **PEBLS** - Example-based symbolic learner
  - Custom exemplar-based classification

### Model Evaluation Metrics
- Precision & Recall
- F1-Score (macro and micro)
- ROC-AUC Score
- Matthews Correlation Coefficient
- Cohen's Kappa
- Confusion Matrix
- Threshold-tuned predictions for precision/recall balance

## Features

- **Custom Transformers**: Clone-safe pipeline components for data preprocessing
- **Feature Engineering**: Temporal patterns, risk groups, and interaction features
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV for model optimization
- **Cross-Validation**: Stratified k-fold validation for robust evaluation
- **Class Imbalance Handling**: Class weights for balanced classification
- **Threshold Optimization**: Fine-tuned decision thresholds for precision-recall trade-offs

## Configuration

All model parameters, feature engineering options, and pipeline settings are centralized in the `Config` dataclass in `classification.py`:

```python
@dataclass
class Config:
    DATA_PATH: str = "clean.csv"
    TARGET_COLUMN: str = "is_violent_crime"
    SAMPLE_SIZE: int = 300_000
    TEST_SIZE: float = 0.20
    # ... and more
```

## Usage

1. **Preprocess data**:
   ```bash
   python preprocessing.py
   ```

2. **Explore data**:
   ```bash
   python eda.py
   ```

3. **Train and evaluate models**:
   ```bash
   python classification.py
   ```

## Output

- **EDA Outputs**: Visualizations and statistical summaries in `eda_outputs/`
- **Results**: Model performance metrics, confusion matrices, and best model details in `results/`
- **Clean Data**: Preprocessed datasets ready for analysis or further modeling

## Technologies

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Utilities**: pathlib, warnings, dataclasses

## Notes

- Data leakage prevention: Target derivation columns (Crime Code) are excluded from model features
- High-cardinality features (location descriptions, IDs) are dropped to improve model generalization
- Models are evaluated on stratified train/validation/test splits for robust performance estimation
