# Heart Disease Prediction - MLOps Pipeline

A complete MLOps implementation for heart disease prediction using K-Nearest Neighbors classifier, featuring FastAPI, Docker containerization, CI/CD with GitHub Actions, and Azure deployment.

## Features

- **Machine Learning Model**: K-Nearest Neighbors classifier trained on heart disease dataset
- **REST API**: FastAPI application with automatic documentation
- **Containerization**: Docker support for consistent deployment
- **CI/CD**: Automated testing and deployment with GitHub Actions
- **Cloud Deployment**: Azure App Service integration

## Prerequisites

- Python 3.9+
- Docker (for containerization)
- Azure account (for deployment)
- GitHub account (for CI/CD)

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will create:
- `model.pkl` - Trained KNN model
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature names for reference

### 3. Run the API Locally

```bash
uvicorn app:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

### 4. Test the API

```bash
# Run all tests
pytest tests/ -v

# Test a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Docker

### Build the Image

```bash
docker build -t heart-disease-api .
```

### Run the Container

```bash
docker run -p 8000:8000 heart-disease-api
```

Access the API at http://localhost:8000

## CI/CD Setup

### GitHub Actions

The project includes two workflows:

1. **CI (Continuous Integration)** - `.github/workflows/ci.yml`
   - Runs on every push and pull request
   - Lints code with flake8
   - Trains model
   - Runs pytest tests

2. **CD (Continuous Deployment)** - `.github/workflows/cd.yml`
   - Runs on push to main branch
   - Builds Docker image
   - Pushes to Azure Container Registry
   - Deploys to Azure App Service

### Required GitHub Secrets

Configure these secrets in your GitHub repository settings:

- `AZURE_CREDENTIALS` - Azure service principal credentials
- `REGISTRY_LOGIN_SERVER` - Azure Container Registry server
- `REGISTRY_USERNAME` - Azure Container Registry username
- `REGISTRY_PASSWORD` - Azure Container Registry password

## Azure Deployment

### 1. Create Azure Resources

```bash
# Create resource group
az group create --name heart-disease-rg --location eastus

# Create container registry
az acr create --resource-group heart-disease-rg \
  --name heartdiseaseacr --sku Basic

# Create app service plan
az appservice plan create --name heart-disease-plan \
  --resource-group heart-disease-rg \
  --is-linux --sku B1

# Create web app
az webapp create --resource-group heart-disease-rg \
  --plan heart-disease-plan \
  --name heart-disease-api \
  --deployment-container-image-name heartdiseaseacr.azurecr.io/heart-disease-api:latest
```

### 2. Configure GitHub Secrets

```bash
# Get Azure credentials
az ad sp create-for-rbac --name "heart-disease-github" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/heart-disease-rg \
  --sdk-auth

# Get ACR credentials
az acr credential show --name heartdiseaseacr
```

Add these to GitHub Secrets.

### 3. Deploy

Push to main branch - deployment happens automatically!

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint

### `POST /predict`
Make a heart disease prediction

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.2345,
  "risk_level": "Low"
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Project Structure

```
Predicting-Heart-Disease/
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI workflow
│       └── cd.yml           # CD workflow
├── tests/
│   └── test_api.py          # API tests
├── app.py                   # FastAPI application
├── train.py                 # Model training script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore           # Docker ignore file
├── heart.csv               # Dataset
└── README.md               # This file
```

## Development

### Code Quality

```bash
# Lint code
flake8 . --max-line-length=127

# Format code
black .
```

### Model Retraining

To retrain the model with new data:

1. Update `heart.csv` with new data
2. Run `python train.py`
3. Test the new model locally
4. Commit and push to trigger CI/CD

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request


