from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
import pickle
import os
import time
import psutil
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import logging
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from prometheus_flask_exporter import PrometheusMetrics

# Configure logging to file
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Static information as metric
metrics.info('app_info', 'Weather Prediction Application', version='1.0.0')

# Define custom metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', [
                        'method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])
REQUEST_IN_PROGRESS = Gauge('http_requests_in_progress',
                            'Number of HTTP requests in progress', ['method', 'endpoint'])

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    'model_inference_time_seconds', 'Model Inference Time in Seconds')
CONFIDENCE_SCORE = Histogram('model_confidence_score', 'Model Confidence Score', buckets=[
                             0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# System metrics
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU Usage in Percent')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory Usage in Bytes')
DISK_USAGE = Gauge('disk_usage_percent', 'Disk Usage in Percent')
NETWORK_RECEIVED = Gauge('network_received_bytes_total',
                         'Network Bytes Received')
NETWORK_SENT = Gauge('network_sent_bytes_total', 'Network Bytes Sent')

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
            input_df['RainToday_coded'] = input_df['RainToday'].map(
                raintoday_map)
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

# Update system metrics every 15 seconds


def update_system_metrics():
    try:
        # Update CPU usage
        CPU_USAGE.set(psutil.cpu_percent())

        # Update memory usage
        mem = psutil.virtual_memory()
        MEMORY_USAGE.set(mem.used)

        # Update disk usage
        disk = psutil.disk_usage('/')
        DISK_USAGE.set(disk.percent)

        # Update network IO
        try:
            net_io = psutil.net_io_counters()
            NETWORK_RECEIVED.set(net_io.bytes_recv)
            NETWORK_SENT.set(net_io.bytes_sent)
        except (AttributeError, OSError) as e:
            # In some environments, network metrics might not be available
            logger.warning(f"Could not update network metrics: {e}")

        logger.debug("System metrics updated successfully")
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")


@app.route('/metrics')
def metrics_endpoint():
    """Endpoint for exposing metrics to Prometheus."""
    # Update system metrics before serving
    update_system_metrics()

    # Return all registered metrics
    return Response(prometheus_client.generate_latest(), mimetype="text/plain")


@app.route('/health', methods=['GET'])
@metrics.counter('health_checks_total', 'Number of health checks')
def health_check():
    """Health check endpoint."""
    status_code = 200 if model is not None and preprocessor is not None else 500

    response = {
        'status': 'healthy' if status_code == 200 else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'timestamp': time.time()
    }

    # Record request in the custom counter
    REQUEST_COUNT.labels('GET', '/health', status_code).inc()

    return jsonify(response), status_code


@app.route('/predict', methods=['POST'])
@metrics.counter('predictions_total', 'Number of predictions')
@metrics.histogram('prediction_latency', 'Prediction latency in seconds')
def predict():
    """Prediction endpoint."""
    start_time = time.time()

    if model is None or preprocessor is None:
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        return jsonify({
            'error': 'Model or preprocessor not loaded'
        }), 500

    try:
        data = request.get_json()
        if not data:
            REQUEST_COUNT.labels('POST', '/predict', 400).inc()
            return jsonify({'error': 'No input data provided'}), 400

        # Preprocess the input data
        preprocess_start = time.time()
        processed_data = preprocess_input(data)
        preprocess_time = time.time() - preprocess_start

        # Measure inference time
        inference_start = time.time()
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1]
        inference_time = time.time() - inference_start

        # Record model metrics
        MODEL_INFERENCE_TIME.observe(inference_time)
        CONFIDENCE_SCORE.observe(probability[0])

        total_time = time.time() - start_time

        # Log the prediction details
        logger.info(
            f"Prediction: {bool(prediction[0])}, Confidence: {probability[0]:.4f}, Inference time: {inference_time:.4f}s")

        # Record the successful request
        REQUEST_COUNT.labels('POST', '/predict', 200).inc()

        # Alert if confidence is low
        if probability[0] < 0.6:
            logger.warning(f"Low confidence prediction: {probability[0]:.4f}")

        return jsonify({
            'prediction': bool(prediction[0]),
            'probability': float(probability[0]),
            'metrics': {
                'inference_time_ms': round(inference_time * 1000, 2),
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'total_time_ms': round(total_time * 1000, 2)
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Update metrics before starting the server
    update_system_metrics()

    # Start metrics update thread
    import threading

    def metrics_updater():
        while True:
            update_system_metrics()
            time.sleep(15)  # Update every 15 seconds

    metrics_thread = threading.Thread(target=metrics_updater, daemon=True)
    metrics_thread.start()

    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
