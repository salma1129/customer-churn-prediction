import pandas as pd
from joblib import load
import os
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Summary, Gauge

# --- 1. MLOps Setup: Paths and Monitoring ---

# Define paths to load the artifacts (relative to the project root)
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
SCALER_FILE = os.path.join(ARTIFACTS_PATH, 'scaler.joblib')
MODEL_FILE = os.path.join(ARTIFACTS_PATH, 'final_model.pkl')

# Define Prometheus Metrics
PREDICTIONS_COUNTER = Counter('churn_predictions_total', 'Total number of churn predictions requested')
CHURN_RATE_GAUGE = Gauge('churn_prediction_rate', 'The last calculated churn rate (0 to 1)')
REQUEST_LATENCY = Summary('api_request_latency_seconds', 'Time spent processing request')

# Start Prometheus HTTP server on a non-default port (e.g., 8001)
start_http_server(8001)

# --- 2. Data Structure (Input Schema) ---

# Define the expected input payload using Pydantic for validation
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --- 3. Model and Scaler Loading ---

# Load model and scaler globally on API startup
try:
    model = load(MODEL_FILE)
    scaler = load(SCALER_FILE)
    print("✅ Model and Scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Artifact file not found: {e}")
    # Critical error, API should not run without artifacts
    raise

# --- 4. Preprocessing Function (Simplified for API) ---

def preprocess_api_data(data: dict) -> pd.DataFrame:
    """Transforms raw JSON input into the format required by the model."""
    
    # 1. Convert input dictionary to DataFrame
    df = pd.DataFrame([data])
    
    # 2. Handle 'TotalCharges' (Convert to numeric and impute if necessary for API robustness)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 3. Binary Encoding (Map to 1/0)
    binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0, 'Male': 1, 'Female': 0}
    
    binary_cols_to_map = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'gender'
    ]
    
    for col in binary_cols_to_map:
        if col in df.columns:
            # Using standard 'int' type for container compatibility
            df[col] = df[col].map(binary_map).astype(int) 

    # 4. One-Hot Encoding (Using get_dummies)
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # 5. Scaling Numerical Features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # 6. Ensure column alignment (The Critical Step)
    expected_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
        'InternetService_Fiber optic', 'InternetService_No', 
        'Contract_One year', 'Contract_Two year', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check'
    ]
    
    # Reindex the incoming data to match the order and presence of expected columns
    X_processed = pd.DataFrame(0, index=df.index, columns=expected_cols)
    for col in df.columns:
        if col in X_processed.columns:
            X_processed[col] = df[col]

    return X_processed

# --- 5. FastAPI Endpoint ---

app = FastAPI(title="Customer Churn Prediction API")

@app.post("/predict")
@REQUEST_LATENCY.time()
# FIX: Removed 'async' keyword to resolve the 'coroutine' object error
def predict(features: CustomerFeatures): 
    """
    Accepts customer features and returns a churn probability prediction (0-1).
    """
    
    # Convert Pydantic model to dictionary
    input_data = features.model_dump()
    
    # Preprocess the data using the saved scaler artifact
    processed_df = preprocess_api_data(input_data)
    
    # Make prediction (probability) using the saved model artifact
    churn_probability = model.predict_proba(processed_df)[:, 1][0]
    
    # Update Prometheus Metrics
    PREDICTIONS_COUNTER.inc()
    CHURN_RATE_GAUGE.set(churn_probability) 
    
    # Format response
    result = {
        "churn_probability": round(churn_probability, 4),
        "prediction": "Yes" if churn_probability >= 0.5 else "No",
        "model_name": type(model).__name__
    }

    return result