# --- Dockerfile ---

# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This speeds up the build if requirements haven't changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# This includes src/, data/, and artifacts/
COPY . .

# Expose the API port (8000 for the main app) and the Prometheus metrics port (8001)
EXPOSE 8000
EXPOSE 8001

# Command to run the FastAPI application using Uvicorn
# The command references the app object inside src/predict_api.py
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]