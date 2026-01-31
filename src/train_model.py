import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def download_dataset(self):
        """Download dataset from Kaggle"""
        print("Downloading dataset...")
        os.system('kaggle datasets download -d parisrohan/credit-score-classification -p ./data --unzip')
        
    def clean_numeric(self, x):
        """Clean numeric columns"""
        try:
            return float(str(x).replace('_', ''))
        except:
            return np.nan
    
    def history_age_to_months(self, x):
        """Convert credit history age to months"""
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
    
    def clean_data(self, df):
        """Clean and preprocess data"""
        # Clean numeric columns
        numeric_cols_to_clean = [
            'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
            'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
            'Monthly_Balance'
        ]
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_numeric)
        
        # Convert credit history age to months
        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Age'] = df['Credit_History_Age'].apply(self.history_age_to_months)
        
        # Drop unnecessary columns
        drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Replace invalid values
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        return df
    
    def encode_data(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Encode all object columns
        object_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in object_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def train(self, data_path='./data/train.csv'):
        """Train the credit scoring model"""
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path, low_memory=False)
        
        # Clean data
        print("Cleaning data...")
        df = self.clean_data(df)
        
        # Encode data
        print("Encoding data...")
        df = self.encode_data(df, fit=True)
        
        # Prepare features and target
        X = df.drop('Credit_Score', axis=1)
        y = df['Credit_Score']
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop NaN values
        X = X.dropna()
        y = y[X.index]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale input
        input_scaled = self.scaler.transform([input_data])
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    
    def save_model(self, filepath='./models/credit_model.pkl'):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='./models/credit_model.pkl'):
        """Load model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Train and save model
    credit_model = CreditScoringModel()
    
    # Download dataset if needed
    if not os.path.exists('./data/train.csv'):
        credit_model.download_dataset()
    
    # Train model
    credit_model.train()
    
    # Save model
    credit_model.save_model()
