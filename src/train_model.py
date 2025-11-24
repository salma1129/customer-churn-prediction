# --- Inside src/train_model.py ---

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump
import os


# Import the necessary function from your preprocessing module
# This ensures that data_preprocessing.py runs (and uses the saved scaler if needed)
from data_preprocessing import preprocess_data 

# Define artifact paths using relative paths
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODEL_FILE = os.path.join(ARTIFACTS_PATH, 'final_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

def train_and_evaluate(X, y, model, model_name):
    """Trains a model, calculates AUC (critical for imbalanced data), and performs cross-validation."""
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    # Predict probabilities for AUC calculation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    # Use cross-validation for a more robust performance estimate
    cv_score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    
    print(f"\n--- Results for {model_name} ---")
    print(f"Test AUC Score: {auc:.4f}")
    print(f"5-Fold Cross-Validation AUC: {cv_score:.4f}")
    
    return model, auc

def run_training_pipeline():
    # 1. Load Data
    print(f"Starting training pipeline. Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Check the path in DATA_PATH.")
        return

    # 2. Preprocess Data
    # is_training=True ensures the scaler is FIT and saved (overwriting the existing one)
    processed_df = preprocess_data(df.copy(), is_training=True) 
    
    # 3. Separate Features and Target
    X = processed_df.drop('Churn', axis=1)
    y = processed_df['Churn']
    
    # 4. Initialize and Compare Models
    # Using liblinear solver for smaller datasets/binary classification
    lr = LogisticRegression(solver='liblinear', random_state=42)
    rf = RandomForestClassifier(random_state=42)
    
    best_auc = 0
    best_model = None
    
    # Train and evaluate Logistic Regression
    lr_model, lr_auc = train_and_evaluate(X, y, lr, "Logistic Regression")
    if lr_auc > best_auc:
        best_auc = lr_auc
        best_model = lr_model

    # Train and evaluate Random Forest
    rf_model, rf_auc = train_and_evaluate(X, y, rf, "Random Forest")
    if rf_auc > best_auc:
        best_auc = rf_auc
        best_model = rf_model

    # 5. Save the Best Model
    if best_model:
        dump(best_model, MODEL_FILE)
        print(f"\n✅ Training complete. Best model ({type(best_model).__name__}) saved with AUC {best_auc:.4f} to {MODEL_FILE}")

if __name__ == '__main__':
    run_training_pipeline()