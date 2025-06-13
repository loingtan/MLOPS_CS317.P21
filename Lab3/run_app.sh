#!/bin/bash
# Run script for the Flask application

# Set environment variables
export FLASK_APP=app/app.py
export FLASK_ENV=development
export PYTHONPATH=.

# Create logs directory
mkdir -p app/logs

# Timestamp for log rotation
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if app is already running
if pgrep -f "python -m flask run" > /dev/null; then
    echo "Flask app is already running. Kill it first if you want to restart."
    exit 1
fi

# Install required packages if needed
echo "Checking for required packages..."
echo "Installing required packages with exact versions specified in requirements.txt..."
pip install --no-cache-dir -r app/requirements.txt

# Force reinstall scikit-learn to ensure correct version
echo "Ensuring scikit-learn 1.6.1 is installed (required for compatibility with pickled models)..."
pip install --no-cache-dir --force-reinstall scikit-learn==1.6.1

# Run FastAPI app with output redirection for logging
echo "Starting the Weather Prediction API..."
echo "The application will be available at http://localhost:5050"
echo "The API documentation will be available at http://localhost:5050/docs"
echo "Metrics will be available at http://localhost:5050/metrics"
echo "Logs will be saved to app/stdout.log and app/stderr.log"
echo "Press Ctrl+C to stop the application"

# Navigate to app directory and run the FastAPI app
cd app
python -m uvicorn app:app --host=0.0.0.0 --port=5050 --reload > stdout.log 2> stderr.log
