# 📞 Customer Churn Prediction System: From Data to Deployment

[cite_start]This project implements an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company[cite: 8]. [cite_start]The system is built using a modularized, two-layer architecture: a **Data Science Layer** for model development and an **MLOps Layer** for API deployment and monitoring using Docker and Prometheus/Grafana[cite: 20, 23, 24].

## 🎯 Project Objective

[cite_start]The primary goal is to build a robust system that predicts whether a customer is likely to churn based on their usage patterns and service history[cite: 10]. [cite_start]This prediction enables proactive engagement campaigns to retain high-value customers[cite: 9].

---

## 🏗️ Architecture & Deliverables

[cite_start]The project is structured into two main layers with clear separation of concerns[cite: 21]:

### Data Science Layer (Completed by Dev 1)
| Deliverable | Status | Description |
| :--- | :--- | :--- |
| `src/data_preprocessing.py` | ✅ Complete | [cite_start]Modular script for cleaning, encoding, and scaling features[cite: 17, 21]. |
| `src/train_model.py` | ✅ Complete | [cite_start]Script for training, comparing (Logistic Regression selected), and optimizing the classifier[cite: 18, 19, 21]. |
| `artifacts/scaler.joblib` | ✅ Artifact | Saved state of the fitted `StandardScaler`. **Critical for consistent API predictions.** |
| `artifacts/final_model.pkl` | ✅ Artifact | Saved binary file containing the final **Logistic Regression** model. |

### MLOps Layer (In Progress by Dev 2)
| Deliverable | Status | Description |
| :--- | :--- | :--- |
| `src/predict_api.py` | 📝 To Do | [cite_start]REST API (FastAPI/Flask) to load artifacts and serve predictions[cite: 22]. |
| `requirements.txt` | 📝 To Do | List of Python dependencies for containerization. |
| `Dockerfile` | 📝 To Do | [cite_start]Containerizes the prediction API service[cite: 23]. |
| `docker-compose.yml` | 📝 To Do | [cite_start]Orchestrates the API, Prometheus, and Grafana containers[cite: 23, 30]. |

---

## ⚙️ Technologies

* [cite_start]**Core Libraries:** Python, `pandas`, `scikit-learn`, `joblib`, `FastAPI` (or Flask), `prometheus_client`[cite: 35].
* [cite_start]**MLOps Stack:** Docker, `docker-compose`[cite: 36].
* [cite_start]**Version Control:** Git/GitHub (using the Feature Branch Workflow)[cite: 37].
* [cite_start]**Data:** Telco Customer Churn Dataset.

---

## 📂 Repository Structure

The project uses a standard layout to ensure Dockerization is straightforward:
## 💻 Execution Guide (Dev 2 Instructions)

This guide assumes the required artifacts (`scaler.joblib` and `final_model.pkl`) have been successfully generated and committed by Dev 1.

### Phase 1: Local Setup & Artifact Generation (Dev 1 Steps)

*(This step has been completed to generate the artifacts)*

1.  **Setup Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install pandas scikit-learn joblib numpy
    ```
2.  **Generate Artifacts:** Run the following two scripts in order:
    ```bash
    python src/data_preprocessing.py # Creates scaler.joblib
    python src/train_model.py         # Creates final_model.pkl
    ```

### Phase 2: MLOps Deployment (Dev 2 Steps)

**Prerequisite:** Docker and Docker Compose must be installed.

1.  **Build the Service Image:**
    *(Dev 2 needs to fill out the `Dockerfile` before running this.)*
    ```bash
    docker-compose build
    ```
2.  **Run the Orchestration:** Start the API, Prometheus, and Grafana services:
    *(Dev 2 needs to fill out the `docker-compose.yml` before running this.)*
    ```bash
    docker-compose up -d
    ```

3.  **Access Services:**
    * **Prediction API:** `http://localhost:8000/predict` (Example endpoint)
    * **Prometheus:** `http://localhost:9090`
    * **Grafana Dashboard:** `http://localhost:3000` (Used to visualize metrics [cite: 24, 32])

---

## 💡 Key Design Decisions (Dev 1 Notes)

1.  **Model Selection:** We compared Logistic Regression and Random Forest. **Logistic Regression** provided the best balance of AUC score and interpretability, making it the final model[cite: 18].
2.  **Imbalance Handling:** Due to the severe class imbalance in the Churn target variable, **AUC (Area Under the ROC Curve)** was chosen as the primary evaluation metric over simple accuracy[cite: 10].
3.  **Scaling:** All numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` to ensure optimal performance for the Logistic Regression model.
