# Startup script to run FastAPI app directly on host and monitoring stack in Docker
# filepath: c:\Users\09398\Subject\Mlops\MLOPS_CS317.P21\Lab3\run_app_without_docker.ps1

# Check if the required Python packages are installed
Write-Host "Checking required Python packages..." -ForegroundColor Cyan
cd app

if (-not (Test-Path -Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found in app directory" -ForegroundColor Red
    exit 1
}

Write-Host "Installing required Python packages..." -ForegroundColor Cyan
pip install --no-cache-dir -r requirements.txt

# Force reinstall scikit-learn to ensure correct version
Write-Host "Ensuring scikit-learn 1.6.1 is installed (required for compatibility with pickled models)..." -ForegroundColor Cyan
pip install --no-cache-dir --force-reinstall scikit-learn==1.6.1

# Start monitoring stack (Prometheus, Grafana, etc.) in Docker
Write-Host "Starting monitoring stack with Docker Compose..." -ForegroundColor Cyan
cd ..
docker-compose up -d prometheus grafana alertmanager node-exporter

# Wait for services to start
Write-Host "Waiting for monitoring services to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Print access information
Write-Host "`nMonitoring stack is now running!" -ForegroundColor Green
Write-Host "Access the following services in your browser:" -ForegroundColor Yellow
Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor Yellow
Write-Host "  - Grafana: http://localhost:3000 (default credentials: admin/admin)" -ForegroundColor Yellow

# Start FastAPI application locally
Write-Host "`nStarting FastAPI application locally..." -ForegroundColor Cyan
Write-Host "The API will be available at http://localhost:5050" -ForegroundColor Yellow
Write-Host "API documentation will be available at http://localhost:5050/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the application`n" -ForegroundColor Yellow

cd app
python -m uvicorn app:app --host 0.0.0.0 --port 5050 --reload
