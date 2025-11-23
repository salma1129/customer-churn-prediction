# --- Inside src/data_preprocessing.py ---

import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load # Need 'load' for the main pipeline to load the data
import numpy as np
import os # To handle saving files in the correct path

# Define the artifact path for saving the scaler
# We save it in the 'artifacts' folder, which is sibling to 'src'
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
SCALER_FILE = os.path.join(ARTIFACTS_PATH, 'scaler.joblib')

# Ensure the artifacts directory exists
os.makedirs(ARTIFACTS_PATH, exist_ok=True)


def handle_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes binary columns (Yes/No, Male/Female) and the target variable."""
    
    # Map Yes/No to 1/0 for all binary columns (including services)
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'PaperlessBilling'
    ]
    # For service columns, 'No internet service' or 'No phone service'
    # is functionally 'No', so we map them to 0 as well.
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]

    mapping = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
    
    # Apply mapping to all binary columns
    for col in binary_cols + service_cols:
        if col in df.columns:
            # Use map and convert to integer type
            df[col] = df[col].map(mapping).astype('Int64', errors='ignore') # Use Int64 for potential nullable int
            
    # Handle 'gender' (Male/Female)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Map the Target Variable (Churn)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Performs One-Hot Encoding on remaining high-cardinality categorical features."""
    
    # These columns have more than two categories or are not strictly Yes/No
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    # Use pandas get_dummies for one-hot encoding
    # drop_first=True prevents multicollinearity
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df

def scale_numerical_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Scales numerical features using StandardScaler.
    Saves the scaler if it's the training phase.
    """
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    if is_training:
        # Initialize and fit the scaler on the training data
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Save the fitted scaler artifact
        dump(scaler, SCALER_FILE) 
        print(f"StandardScaler fitted and saved to: {SCALER_FILE}")
    else:
        # Load the saved scaler for consistent transformation (in the API/testing phase)
        try:
            scaler = load(SCALER_FILE)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at {SCALER_FILE}. Please run training first.")
    
    return df

def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Runs the full preprocessing pipeline."""
    
    # --- 1. Initial Cleaning and Type Conversion (Safe-guard against common issue) ---
    # The 'TotalCharges' column can sometimes contain spaces (' '). We handle this robustly.
    df = df[df['TotalCharges'] != ' '].copy()
    # Convert to numeric, errors='coerce' turns non-convertible values into NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop any remaining NaNs (if any non-space issue remains)
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # --- 2. Drop the identifier column ---
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # --- 3. Feature Engineering Steps ---
    df = handle_binary_features(df)
    df = encode_categorical_features(df)
    
    # --- 4. Scaling Numerical features ---
    df = scale_numerical_features(df, is_training=is_training)
    
    return df

if __name__ == '__main__':
    # Block for local testing of the preprocessing script
    
    # Assume the raw data is in the 'data' folder
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print(f"Loading data from: {data_path}")
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Make sure it's in the '../data/' folder.")
        exit()

    # Create a copy to prevent modifying the original DataFrame
    processed_df = preprocess_data(raw_df.copy(), is_training=True) 
    
    # Verification prints
    print("\n--- Preprocessing Test Complete ---")
    print(f"Final shape: {processed_df.shape}")
    print("First 5 rows of processed data:")
    print(processed_df.head())
    print("\nData types (confirming all are numeric):")
    print(processed_df.dtypes)
    print("Columns:")
    print(processed_df.columns.tolist())