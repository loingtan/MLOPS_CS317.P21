# Set environment variables
$env:FLASK_APP = "app/app.py"
$env:FLASK_ENV = "development"
$env:PYTHONPATH = "."

# Create directory for logs
if (-not (Test-Path -Path "app\logs")) {
    New-Item -ItemType Directory -Path "app\logs"
}

# Run Flask app with stdout and stderr redirection for logging
Write-Host "Starting the Weather Prediction API..."
Write-Host "The application will be available at http://localhost:5000"
Write-Host "Press Ctrl+C to stop the application"

# Start the Flask app and redirect output to log files
python -m flask run --host=0.0.0.0 --port=5000 2>&1 | Tee-Object -FilePath "app\stderr.log" *>&1 | Tee-Object -FilePath "app\stdout.log"
