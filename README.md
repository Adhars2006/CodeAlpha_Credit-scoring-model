# CodeAlpha_Credit-scoring-model

A credit scoring pipeline that downloads a Kaggle credit-score dataset, performs cleaning and feature engineering, encodes categorical variables, scales numeric features, trains several classifiers (Logistic Regression, Decision Tree, Random Forest), and prints evaluation metrics and ROC plots. The repository includes a minimal README; this file expands usage and reproducibility instructions.

## Table of Contents

- [Purpose](#purpose)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
- [Results & Visualization](#results--visualization)
- [Customizing / Extending](#customizing--extending)
- [Contributing](#contributing)
- [License](#license)

## Purpose

This project demonstrates a full ML pipeline for credit scoring:
- Data cleaning (numeric cleaning, handling 'Credit_History_Age')
- Categorical encoding (LabelEncoder)
- Feature scaling (StandardScaler)
- Model training and comparison with multiple classifiers
- Displaying common metrics (Precision, Recall, F1, ROC-AUC) and ROC curve plotting

## Requirements

- Python 3.8+
- Packages:
  - pandas, numpy
  - scikit-learn
  - matplotlib
  - kaggle (if using the script's download step)

Install example:

pip install pandas numpy scikit-learn matplotlib kaggle

## Dataset

model.py attempts to download the dataset via Kaggle:

kaggle datasets download -d parisrohan/credit-score-classification -p ./dataset --unzip

Alternatively, download manually and place `train.csv` (and test.csv if available) under `./dataset/`.

## Installation

1. Clone repository:

git clone https://github.com/Adhars2006/CodeAlpha_Credit-scoring-model.git
cd CodeAlpha_Credit-scoring-model

2. Install dependencies:

pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn matplotlib kaggle

3. Configure Kaggle credentials or place dataset in ./dataset/.

## Usage

Run the main pipeline:

python model.py

The script will:
- Download (or read) `train.csv`
- Clean numeric columns and convert `Credit_History_Age` to months
- Encode and handle missing values
- Create X, y (the script converts `Credit_Score` into a binary label: Poor=1, others=0 — see model.py)
- Train Logistic Regression, Decision Tree, Random Forest
- Print metrics for each model and optionally plot ROC curve for Random Forest

## Pipeline Details

- Cleaning helpers: `clean_numeric`, `history_age_to_months`
- Columns automatically dropped if present: ID-like fields (ID, Customer_ID, Month, Name, SSN)
- Categorical columns encoded with LabelEncoder; missing values filled with mode or median depending on type
- Numeric features scaled with StandardScaler

## Results & Visualization

- Printed metrics per model (Precision, Recall, F1, ROC-AUC)
- If `predict_proba` exists for a model, script computes ROC-AUC and can plot ROC curve (matplotlib)

## Customizing / Extending

- Change target: model.py currently binarizes `Credit_Score` to Poor vs Not-Poor. Modify if you want multi-class classification.
- Add more models (XGBoost, LightGBM) and cross-validation (GridSearchCV).
- Add feature importance plots and permutation importance.
- Save model artifacts (joblib / pickle) and add a simple inference script.

## Contributing

Open issues and PRs are welcome. Please include a description of the change and relevant tests or sample outputs.

## License

Add a LICENSE file to define the intended license (e.g., MIT).