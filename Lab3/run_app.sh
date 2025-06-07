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

# Run Flask app with output redirection for logging
echo "Starting the Weather Prediction API..."
echo "The application will be available at http://localhost:5000"
echo "Prometheus metrics will be available at http://localhost:5000/metrics"
echo "Logs will be saved to app/stdout.log and app/stderr.log"
echo "Press Ctrl+C to stop the application"

# Run Flask app and redirect output to log files
python -m flask run --host=0.0.0.0 --port=5000 > app/stdout.log 2> app/stderr.log
