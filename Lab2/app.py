from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'preprocessing', 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info(f"Preprocessor loaded successfully from {PREPROCESSOR_PATH}")
except Exception as e:
    logger.error(f"Error loading preprocessor: {e}")
    preprocessor = None

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
            input_df['RainToday_coded'] = input_df['RainToday'].map(raintoday_map)
            input_df = input_df.drop(columns=['RainToday'])
        
        # Drop Date and Location columns if present
        if 'Date' in input_df.columns:
            input_df = input_df.drop(columns=['Date'])
        if 'Location' in input_df.columns:
            input_df = input_df.drop(columns=['Location'])
        
        # Transform the input data using the loaded preprocessor
        processed_data = preprocessor.transform(input_df)
        
        return processed_data
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model or preprocessor not loaded'
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Preprocess the input data
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1]

        return jsonify({
            'prediction': str(prediction[0]),
            'probability': float(probability[0])
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 