"""
Heart Disease Prediction API

FastAPI application that serves the trained heart disease prediction model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Literal


# Load model and scaler at startup
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("Model, scaler, and feature names loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    scaler = None
    feature_names = None


# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using K-Nearest Neighbors classifier",
    version="1.0.0"
)


# Pydantic model for input validation
class PatientData(BaseModel):
    Age: int = Field(..., ge=0, le=120, description="Age in years")
    Sex: Literal["M", "F"] = Field(..., description="Sex: M (Male) or F (Female)")
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(
        ..., 
        description="Chest pain type: TA (Typical Angina), ATA (Atypical Angina), NAP (Non-Anginal Pain), ASY (Asymptomatic)"
    )
    RestingBP: int = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=0, description="Serum cholesterol (mm/dl)")
    FastingBS: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(
        ..., 
        description="Resting ECG results: Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy)"
    )
    MaxHR: int = Field(..., ge=60, le=202, description="Maximum heart rate achieved")
    ExerciseAngina: Literal["Y", "N"] = Field(..., description="Exercise-induced angina: Y (Yes) or N (No)")
    Oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(
        ..., 
        description="Slope of peak exercise ST segment: Up (upsloping), Flat, Down (downsloping)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "Age": 40,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 140,
                "Cholesterol": 289,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 172,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class: 0 (No heart disease) or 1 (Heart disease)")
    probability: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make a prediction (POST)",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease for a patient.
    
    Returns:
        - prediction: 0 (No heart disease) or 1 (Heart disease)
        - probability: Probability of heart disease (0-1)
        - risk_level: Low (<0.3), Medium (0.3-0.7), or High (>0.7)
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = patient.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables (same as training)
        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
        
        # Ensure all expected features are present
        for feature in feature_names:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        
        # Get probability (if available)
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(input_scaled)[0][1])
        else:
            # For KNN, we can use the neighbors to estimate probability
            neighbors = model.kneighbors(input_scaled, return_distance=False)[0]
            y_train = model._y  # Access training labels
            neighbor_labels = [y_train[i] for i in neighbors]
            probability = sum(neighbor_labels) / len(neighbor_labels)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
