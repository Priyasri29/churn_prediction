# main.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd
from model import train_model, load_model, save_model
from data_utils import preprocess_data, load_data
from explanations import generate_shap_explanation, generate_lime_explanation

app = FastAPI()

# Load or train model
model_path = "model.pkl"

@app.post("/train")
async def train_model_endpoint(file: UploadFile = File(...)):
    """Endpoint to upload data and train the model."""
    data = pd.read_csv(file.file)
    preprocessed_data, _ = preprocess_data(data)
    model = train_model(preprocessed_data, _)
    save_model(model, model_path)
    return {"message": "Model trained and saved successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict churn on uploaded dataset."""
    model = load_model(model_path)
    data = pd.read_csv(file.file)
    preprocessed_data, _ = preprocess_data(data)
    predictions = model.predict(preprocessed_data.drop('churn', axis=1))
    return {"predictions": predictions.tolist()}

@app.post("/explain/shap")
async def explain_shap(file: UploadFile = File(...)):
    """Explain predictions using SHAP values."""
    model = load_model(model_path)
    data = pd.read_csv(file.file)
    preprocessed_data, _ = preprocess_data(data)
    shap_values = generate_shap_explanation(model, preprocessed_data.drop('churn', axis=1))
    return {"shap_values": shap_values.tolist()}

@app.post("/explain/lime")
async def explain_lime(file: UploadFile = File(...), instance_idx: int = 0):
    """Explain predictions using LIME for a specific instance."""
    model = load_model(model_path)
    data = pd.read_csv(file.file)
    preprocessed_data, _ = preprocess_data(data)
    instance = preprocessed_data.drop('churn', axis=1).iloc[instance_idx]
    lime_explanation = generate_lime_explanation(model, preprocessed_data.drop('churn', axis=1), instance)
    return {"lime_explanation": lime_explanation.as_list()}
