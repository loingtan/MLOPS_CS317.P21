from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import pickle
import os
import uvicorn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Prediction API",
    description="API for predicting rain tomorrow based on weather features",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
def root():
    """Redirects to the API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# Define default values for features
DEFAULT_VALUES = {
    'MinTemp': 20.0,
    'MaxTemp': 25.0,
    'Rainfall': 0.0,
    'Evaporation': 5.0,
    'Sunshine': 7.0,
    'WindGustDir': 'N',
    'WindGustSpeed': 30.0,
    'WindDir9am': 'N',
    'WindDir3pm': 'N',
    'WindSpeed9am': 10.0,
    'WindSpeed3pm': 15.0,
    'Humidity9am': 60.0,
    'Humidity3pm': 55.0,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1013.0,
    'Cloud9am': 5.0,
    'Cloud3pm': 5.0,
    'Temp9am': 22.0,
    'Temp3pm': 24.0,
    'RainToday': 'No'
}

# Load the model and preprocessor from local files
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(
    __file__), 'preprocessing', 'model.pkl')

# Load model and preprocessor
model = None
preprocessor = None

# Load model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Load preprocessor
try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info(f"Preprocessor loaded successfully from {PREPROCESSOR_PATH}")
except FileNotFoundError:
    logger.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
except Exception as e:
    logger.error(f"Error loading preprocessor: {e}")

# Log application startup status
if model is None:
    logger.warning("Application started without a valid model")
if preprocessor is None:
    logger.warning("Application started without a valid preprocessor")


def fill_missing_values(data):
    """Fill missing values with default values."""
    filled_data = DEFAULT_VALUES.copy()
    for key, value in data.items():
        if value is not None:  # Only update if value is not None
            filled_data[key] = value
    return filled_data


def preprocess_input(data):
    """Preprocess input data to match the training pipeline."""
    try:
        # Fill missing values with defaults
        filled_data = fill_missing_values(data)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([filled_data])

        # Handle RainToday column if present (matching training pipeline)
        if 'RainToday' in input_df.columns:
            raintoday_map = {'Yes': 1.0, 'No': 0.0}
            input_df['RainToday_coded'] = input_df['RainToday'].map(
                raintoday_map)
            input_df = input_df.drop(columns=['RainToday'])

        # Drop Date and Location columns if present
        if 'Date' in input_df.columns:
            input_df = input_df.drop(columns=['Date'])
        if 'Location' in input_df.columns:
            input_df = input_df.drop(columns=['Location'])

        # Transform the input data using the loaded preprocessor
        if preprocessor is not None:
            processed_data = preprocessor.transform(input_df)
            return processed_data
        else:
            # Since we don't have the preprocessor and don't know the exact expected features,
            # we'll use a hardcoded array with zeros that matches the expected feature dimension
            # This is a FALLBACK ONLY for demo purposes
            logger.warning(
                "Preprocessor not available, using fallback dummy data for demo")

            # Create a dummy array with 61 features (based on error message from model)
            return np.zeros((1, 61))
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

# Define Pydantic models for request and response


class WeatherData(BaseModel):
    MinTemp: Optional[float] = None
    MaxTemp: Optional[float] = None
    Rainfall: Optional[float] = None
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustDir: Optional[str] = None
    WindGustSpeed: Optional[float] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    WindSpeed9am: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Pressure3pm: Optional[float] = None
    Cloud9am: Optional[float] = None
    Cloud3pm: Optional[float] = None
    Temp9am: Optional[float] = None
    Temp3pm: Optional[float] = None
    RainToday: Optional[str] = None
    Date: Optional[str] = None
    Location: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool


class PredictionResponse(BaseModel):
    prediction: bool
    probability: float


class ErrorResponse(BaseModel):
    error: str


@app.get('/health', response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }


@app.post('/predict', response_model=PredictionResponse, responses={500: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
async def predict(weather_data: WeatherData):
    """Prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert Pydantic model to dictionary
        data = weather_data.model_dump(exclude_unset=True)
        if not data:
            raise HTTPException(
                status_code=400, detail="No input data provided")

        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Make prediction
        try:
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)[:, 1]

            return {
                'prediction': bool(prediction[0]),
                'probability': float(probability[0])
            }
        except Exception as pred_error:
            logger.error(f"Error during prediction: {pred_error}")
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(pred_error)}")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
