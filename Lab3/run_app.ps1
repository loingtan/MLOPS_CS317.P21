# Set environment variables
$env:PYTHONPATH = "."

# Create directory for logs
if (-not (Test-Path -Path "app\logs")) {
    New-Item -ItemType Directory -Path "app\logs"
}

# Run FastAPI app with stdout and stderr redirection for logging
Write-Host "Starting the Weather Prediction API..."
Write-Host "The application will be available at http://localhost:5050"
Write-Host "The API documentation will be available at http://localhost:5050/docs"
Write-Host "Press Ctrl+C to stop the application"

# Install required packages if needed
Write-Host "Checking for required packages..."
Write-Host "Installing required packages with exact versions specified in requirements.txt..."
pip install --no-cache-dir -r app/requirements.txt

# Force reinstall scikit-learn to ensure correct version
Write-Host "Ensuring scikit-learn 1.6.1 is installed (required for compatibility with pickled models)..."
pip install --no-cache-dir --force-reinstall scikit-learn==1.6.1

# Start the FastAPI app with Uvicorn and redirect output to log files
cd app
python -m uvicorn app:app --host=0.0.0.0 --port=5050 --reload 2>&1 | Tee-Object -FilePath "stderr.log" *>&1 | Tee-Object -FilePath "stdout.log"
