# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. DATA LOADING & CLEANING
# ======================

def load_and_clean_data(file_path):
    """Load and clean the credit card default dataset"""
    df = pd.read_excel(file_path, header=1)  # Skip first row with X1, X2 labels
    
    # Rename columns for clarity
    df = df.rename(columns={
        'PAY_0': 'PAY_1',  # For consistent naming (PAY_1 to PAY_6)
        'default payment next month': 'DEFAULT'
    })
    
    # Handle missing values (if any)
    df.dropna(subset=['DEFAULT'], inplace=True)
    
    # Clean categorical variables
    df = df[(df['SEX'].isin([1, 2])) & 
           (df['EDUCATION'].isin([1, 2, 3, 4])) & 
           (df['MARRIAGE'].isin([1, 2, 3]))]
    
    # Clean payment status (-2 = no consumption, -1 = paid duly, 0 = unknown, 1-9 = delays)
    pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for col in pay_columns:
        df = df[df[col].isin(range(-2, 10))]
    
    # Clean bill and payment amounts
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    
    for col in bill_cols + pay_amt_cols:
        df[col] = df[col].clip(lower=0)
    
    return df

# ======================
# 2. FEATURE ENGINEERING
# ======================

def create_features(df):
    """Create new features from existing data"""
    # Payment behavior features
    pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['AVG_PAY_DELAY'] = df[pay_columns].apply(lambda x: x[x > 0].mean(), axis=1).fillna(0)
    df['MAX_PAY_DELAY'] = df[pay_columns].max(axis=1)
    df['PAY_DELAY_COUNT'] = df[pay_columns].apply(lambda x: sum(x > 0), axis=1)
    
    # Balance utilization
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['AVG_BILL_RATIO'] = df[bill_cols].mean(axis=1) / df['LIMIT_BAL']
    
    # Payment patterns
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['PAYMENT_RATIO'] = df[pay_amt_cols].sum(axis=1) / (df[bill_cols].sum(axis=1) + 1e-6)
    
    # Age groups
    df['AGE_GROUP'] = pd.cut(df['AGE'], 
                            bins=[20, 30, 40, 50, 60, 70, 80],
                            labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    return df

# ======================
# 3. DATA PREPARATION
# ======================

def prepare_data(df):
    """Prepare data for modeling"""
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE', 'AGE_GROUP'], drop_first=True)
    
    # Separate features and target
    X = df.drop(['ID', 'DEFAULT'], axis=1)
    y = df['DEFAULT']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale numerical features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test

# ======================
# 4. MODEL TRAINING
# ======================

def train_models(X_train, y_train):
    """Train and evaluate multiple models"""
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        # Full training
        model.fit(X_train, y_train)
        
        results[name] = {
            'model': model,
            'cv_mean_f1': np.mean(cv_scores),
            'cv_std_f1': np.std(cv_scores)
        }
    
    return results

# ======================
# 5. EVALUATION
# ======================

def evaluate_models(results, X_test, y_test):
    """Evaluate models on test set"""
    evaluation = {}
    for name, result in results.items():
        y_pred = result['model'].predict(X_test)
        
        evaluation[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'cv_mean_f1': result['cv_mean_f1'],
            'cv_std_f1': result['cv_std_f1']
        }
    
    return evaluation

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    file_path = r"C:\Users\ismail\Desktop\Calisma\3rd Grade Study Files\COM3549-FINTECH\Fintech Project Files\Credit_Card_Clients_dataset.xls"
    # 1. Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)
    
    # 2. Feature engineering
    print("Creating new features...")
    df = create_features(df)
    
    # 3. Prepare data for modeling
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 4. Train models
    print("Training models...")
    models = train_models(X_train, y_train)
    
    # 5. Evaluate models
    print("Evaluating models...")
    evaluation = evaluate_models(models, X_test, y_test)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for name, result in evaluation.items():
        print(f"\n{name}:")
        print(f"Cross-Validation F1: {result['cv_mean_f1']:.3f} ± {result['cv_std_f1']:.3f}")
        print("\nClassification Report:")
        print(result['classification_report'])
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])
