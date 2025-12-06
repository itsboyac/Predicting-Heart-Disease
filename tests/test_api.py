"""
Unit tests for the Heart Disease Prediction API
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert data["scaler_loaded"] == True


def test_predict_endpoint_valid_input():
    """Test prediction with valid input data."""
    patient_data = {
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
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_endpoint_high_risk():
    """Test prediction with high-risk patient data."""
    patient_data = {
        "Age": 65,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 300,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 100,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Down"
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data


def test_predict_endpoint_invalid_sex():
    """Test prediction with invalid sex value."""
    patient_data = {
        "Age": 40,
        "Sex": "X",  # Invalid
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
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_age():
    """Test prediction with invalid age value."""
    patient_data = {
        "Age": 150,  # Invalid (too high)
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
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_missing_field():
    """Test prediction with missing required field."""
    patient_data = {
        "Age": 40,
        "Sex": "M",
        # Missing ChestPainType
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
