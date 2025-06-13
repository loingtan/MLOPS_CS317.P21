# Test script for the Weather Prediction API Monitoring Demo (Windows PowerShell version)

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "         Weather Prediction API Monitoring & Logging Test" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if the API is running
Write-Host "Checking if the API is running..."
try {
    Invoke-WebRequest -Uri "http://localhost:5050/health" -Method Get -UseBasicParsing | Out-Null
    Write-Host "✓ API is running" -ForegroundColor Green
} catch {
    Write-Host "✗ API is not running. Please start the API first with ./run_app.ps1" -ForegroundColor Red
    Write-Host "Exiting..."
    exit 1
}

# Check if monitoring stack is running
Write-Host "Checking if monitoring stack is running..."
try {
    Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -Method Get -UseBasicParsing | Out-Null
    Write-Host "✓ Prometheus is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Prometheus is not running. Please start the monitoring stack with:" -ForegroundColor Red
    Write-Host "docker-compose up -d" -ForegroundColor Yellow
    Write-Host "Exiting..."
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "                      Running API Tests" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

Write-Host "Running API tests..."
python test_api.py

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Test complete! You can now check the monitoring dashboards:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Grafana: http://localhost:3000 (login: admin/admin)" -ForegroundColor Yellow
Write-Host "🔍 Prometheus: http://localhost:9090" -ForegroundColor Yellow
Write-Host "🚨 AlertManager: http://localhost:9093" -ForegroundColor Yellow
Write-Host "📝 Fluent Bit: http://localhost:2020" -ForegroundColor Yellow
Write-Host ""
Write-Host "To see the complete metrics from the API, visit:" -ForegroundColor Yellow
Write-Host "http://localhost:5050/metrics" -ForegroundColor Yellow
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
