# Error simulation script for testing the alerting functionality (Windows PowerShell version)

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "         Error Simulation for Alerting System Testing" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to generate error requests
function Generate-Errors {
    param (
        [int]$numErrors,
        [int]$delay
    )
    
    Write-Host "Generating $numErrors error requests with $delay second delay between requests..." -ForegroundColor Yellow
    
    for ($i = 1; $i -le $numErrors; $i++) {
        # Send an invalid request to trigger an error
        Write-Host -NoNewline "Sending error request $i/$numErrors... "
        
        # Invalid JSON request that will cause an error
        try {
            Invoke-WebRequest -Uri "http://localhost:5050/predict" `
                -Method Post `
                -ContentType "application/json" `
                -Body '{"MinTemp": "invalid", "MaxTemp": "NaN", "invalidField": true}' `
                -UseBasicParsing | Out-Null
        } catch {
            # We expect an error, so this is fine
        }
            
        Write-Host "Done." -ForegroundColor Green
        # Wait for specified delay
        Start-Sleep -Seconds $delay
    }
}

# Check if API is running
Write-Host "Checking if the API is running..."
try {
    Invoke-WebRequest -Uri "http://localhost:5050/health" -Method Get -UseBasicParsing | Out-Null
    Write-Host "✓ API is running" -ForegroundColor Green
} catch {
    Write-Host "✗ API is not running. Please start the API first with ./run_app.ps1" -ForegroundColor Red
    Write-Host "Exiting..."
    exit 1
}

# Check if monitoring stack is running (Prometheus)
Write-Host "Checking if monitoring stack is running..."
try {
    Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -Method Get -UseBasicParsing | Out-Null
    Write-Host "✓ Monitoring stack is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Prometheus is not running. Please start the monitoring stack with:" -ForegroundColor Red
    Write-Host "docker-compose up -d" -ForegroundColor Yellow
    Write-Host "Exiting..."
    exit 1
}

# Run test scenarios
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "                  Running Error Test Scenarios" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# Scenario 1: High error rate
Write-Host "`nScenario 1: Generating high error rate to trigger alert" -ForegroundColor Yellow
Generate-Errors -numErrors 30 -delay 1

Write-Host "`nTest complete!" -ForegroundColor Green
Write-Host "`nCheck the Grafana dashboard to see the error rate spike:" -ForegroundColor Yellow
Write-Host "http://localhost:3000/d/weather/weather-prediction-api-dashboard" -ForegroundColor Cyan
