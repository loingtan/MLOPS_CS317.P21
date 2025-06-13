import threading
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List
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
from prometheus_client import Counter, Histogram, Gauge, Info, Summary, REGISTRY
from prometheus_client.registry import CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextlib import asynccontextmanager

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

# Create a custom registry to handle reloads
# Check if registry already exists in globals to handle uvicorn reload
if "metrics_registry" not in globals():
    metrics_registry = CollectorRegistry()
    # Register default collectors for this registry
    try:
        # Only use these if available
        prometheus_client.gc_collector.GCCollector(metrics_registry)
        prometheus_client.platform_collector.PlatformCollector(
            metrics_registry)
    except Exception as e:
        logger.warning(f"Could not register default collectors: {e}")
else:
    metrics_registry = globals()["metrics_registry"]

# Define Prometheus metrics middleware


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        # These will be initialized after the get_or_create_metric function is defined
        self.request_count = None
        self.request_latency = None
        self.request_in_progress = None
        self.initialized = False

    async def dispatch(self, request: Request, call_next):
        # Initialize metrics if not initialized
        if not self.initialized:
            self.request_count = get_or_create_metric(
                Counter, 'http_requests_total', 'Total HTTP Requests',
                labelnames=['method', 'endpoint', 'status'])
            self.request_latency = get_or_create_metric(
                Histogram, 'http_request_duration_seconds', 'HTTP Request Latency',
                labelnames=['method', 'endpoint'])
            self.request_in_progress = get_or_create_metric(
                Gauge, 'http_requests_in_progress', 'Number of HTTP requests in progress',
                labelnames=['method', 'endpoint'])
            self.initialized = True

        start_time = time.time()
        path = request.url.path
        method = request.method

        # Increase in-progress metric
        self.request_in_progress.labels(method=method, endpoint=path).inc()

        # Process the request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            # Decrease in-progress metric
            self.request_in_progress.labels(method=method, endpoint=path).dec()

            # Observe latency
            self.request_latency.labels(method=method, endpoint=path).observe(
                time.time() - start_time)

            # Count request
            self.request_count.labels(
                method=method, endpoint=path, status=status_code).inc()

        return response

# Create a function to safely get or create metrics


def get_or_create_metric(metric_type, name, documentation, **kwargs):
    try:
        # Try to get the existing metric from our custom registry
        if name in metrics_registry._names_to_collectors:
            return metrics_registry._names_to_collectors[name]
        # If it doesn't exist, create a new one with our registry
        return metric_type(name, documentation, registry=metrics_registry, **kwargs)
    except ValueError as e:
        logger.warning(f"Error creating metric {name}: {e}")
        # If the metric already exists but we couldn't get it directly, search for it
        for metric in metrics_registry.collect():
            if metric.name == name:
                return metrics_registry._names_to_collectors[name]
        # If we get here, something else went wrong
        raise


# Static information as metric
APP_INFO = get_or_create_metric(
    Info, 'app_info', 'Weather Prediction Application')
APP_INFO.info({'version': '1.0.0'})

# Model metrics
MODEL_INFERENCE_TIME = get_or_create_metric(
    Histogram, 'model_inference_time_seconds', 'Model Inference Time in Seconds')
CONFIDENCE_SCORE = get_or_create_metric(
    Histogram, 'model_confidence_score', 'Model Confidence Score',
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# System metrics
CPU_USAGE = get_or_create_metric(
    Gauge, 'cpu_usage_percent', 'CPU Usage in Percent')
MEMORY_USAGE = get_or_create_metric(
    Gauge, 'memory_usage_bytes', 'Memory Usage in Bytes')
DISK_USAGE = get_or_create_metric(
    Gauge, 'disk_usage_percent', 'Disk Usage in Percent')
NETWORK_RECEIVED = get_or_create_metric(
    Gauge, 'network_received_bytes_total', 'Network Bytes Received')
NETWORK_SENT = get_or_create_metric(
    Gauge, 'network_sent_bytes_total', 'Network Bytes Sent')

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


class MetricsData(BaseModel):
    inference_time_ms: float
    preprocess_time_ms: float
    total_time_ms: float


class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    metrics: MetricsData


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    timestamp: float


class ErrorResponse(BaseModel):
    error: str


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

# Initialize model and preprocessor
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
            # If preprocessor is not available, perform basic preprocessing
            # This is a fallback and won't work correctly in production
            logger.warning(
                "Preprocessor not available, using fallback dummy data")

            # Create a dummy array with 61 features (based on expected model input)
            return np.zeros((1, 61))
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

# Create metrics update thread


def metrics_updater():
    while True:
        update_system_metrics()
        time.sleep(15)  # Update every 15 seconds


# Create metrics thread
metrics_thread = threading.Thread(target=metrics_updater, daemon=True)

# Initialize the metrics on startup
update_system_metrics()

# Define lifespan for proper startup/shutdown handling


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup operations
    logger.info("Starting application")
    # Start the metrics thread on startup
    metrics_thread.start()
    logger.info("Metrics update thread started")
    yield
    # Shutdown operations
    logger.info("Application shutting down")

# Create the FastAPI app
app = FastAPI(
    title="Weather Prediction API",
    description="API for predicting rain tomorrow based on weather features",
    version="1.0.0",
    lifespan=lifespan
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects to the API documentation."""
    return RedirectResponse(url="/docs")


@app.get('/metrics', include_in_schema=False)
async def metrics_endpoint():
    """Endpoint for exposing metrics to Prometheus."""
    # Update system metrics before serving
    update_system_metrics()

    # Return metrics from our custom registry
    return Response(
        content=prometheus_client.generate_latest(metrics_registry),
        media_type="text/plain"
    )


@app.get('/health', response_model=HealthResponse, responses={500: {"model": ErrorResponse}})
async def health_check():
    """Health check endpoint."""
    status_code = 200 if model is not None and preprocessor is not None else 500

    response = {
        'status': 'healthy' if status_code == 200 else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'timestamp': time.time()
    }

    if status_code != 200:
        raise HTTPException(status_code=status_code,
                            detail="Service unhealthy")

    return response


@app.post('/predict', response_model=PredictionResponse, responses={500: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
async def predict(weather_data: WeatherData):
    """Prediction endpoint."""
    start_time = time.time()

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert Pydantic model to dictionary
        data = weather_data.model_dump(exclude_unset=True)
        if not data:
            raise HTTPException(
                status_code=400, detail="No input data provided")

        # Preprocess the input data
        preprocess_start = time.time()
        processed_data = preprocess_input(data)
        preprocess_time = time.time() - preprocess_start

        # Measure inference time
        inference_start = time.time()
        try:
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

            # Alert if confidence is low
            if probability[0] < 0.6:
                logger.warning(
                    f"Low confidence prediction: {probability[0]:.4f}")

            return {
                'prediction': bool(prediction[0]),
                'probability': float(probability[0]),
                'metrics': {
                    'inference_time_ms': round(inference_time * 1000, 2),
                    'preprocess_time_ms': round(preprocess_time * 1000, 2),
                    'total_time_ms': round(total_time * 1000, 2)
                }
            }
        except Exception as pred_error:
            logger.error(f"Error during prediction: {pred_error}")
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(pred_error)}")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5050, reload=True)
# Added another test comment to trigger reload
