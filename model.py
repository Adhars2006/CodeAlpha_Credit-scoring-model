import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Download the dataset from Kaggle
# Note: You need to have Kaggle API installed and authenticated.
# Install kaggle if not: pip install kaggle
# Place your kaggle.json in ~/.kaggle/
os.system('kaggle datasets download -d parisrohan/credit-score-classification -p ./dataset --unzip')

# The dataset has train.csv and test.csv; we'll use train.csv which has the labels

# Step 2: Load the data
data_path = './dataset/train.csv'
df = pd.read_csv(data_path)

# Step 3: Data Cleaning and Feature Engineering
# From dataset info, some columns need cleaning: remove non-numeric characters, handle missing values, etc.

# Define function to clean numeric columns (remove '_' or other artifacts)
def clean_numeric(x):
    try:
        return float(str(x).replace('_', ''))
    except:
        return np.nan

# Columns that need cleaning
numeric_cols_to_clean = [
    'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
    'Monthly_Balance'
]

for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Handle Credit_History_Age: convert to months
def history_age_to_months(x):
    if pd.isna(x):
        return np.nan
    try:
        years, months = 0, 0
        parts = str(x).split()
        if 'Years' in parts:
            years = int(parts[0])
        if 'Months' in parts:
            months = int(parts[3])
        return years * 12 + months
    except:
        return np.nan

if 'Credit_History_Age' in df.columns:
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(history_age_to_months)

# Drop unnecessary columns
drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Handle categorical columns
categorical_cols = [
    'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
]

# Replace invalid values
df['Occupation'] = df['Occupation'].replace('_______', np.nan)
df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    else:
        # Only use median for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

# Encode categorical variables
label_encoders = {}

# First, identify all columns with object dtype
object_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode all object columns
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target variable: Credit_Score (0: Good, 1: Poor, 2: Standard) - but for binary, let's make Poor=1 (bad), others=0 (good)
# But user said classification, and creditworthiness, so multi-class or binary? Approach says classification, metrics for binary but can extend.
# For simplicity, treat as multi-class, but ROC-AUC for multi needs adjustment. Alternatively, make binary: Poor vs not Poor.

# Let's make binary: bad (Poor) vs good (Standard/Good)
df['Credit_Score'] = np.where(df['Credit_Score'] == label_encoders['Credit_Score'].transform(['Poor'])[0], 1, 0)

# Features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Ensure all features are numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop any rows with NaN values that might have been created
X = X.dropna()
y = y[X.index]

# Scale numeric features
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    results[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# Step 6: Display results
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value else f"{metric}: N/A")

# Optional: Plot ROC for one model, e.g., Random Forest
if 'Random Forest' in models:
    rf_prob = models['Random Forest'].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, rf_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
