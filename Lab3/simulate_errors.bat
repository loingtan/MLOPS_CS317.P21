@echo off
REM Error simulation script for testing the alerting functionality (Windows batch version)

echo ===================================================================
echo          Error Simulation for Alerting System Testing
echo ===================================================================
echo.

REM Check if API is running
echo Checking if the API is running...
curl -s http://localhost:5050/health > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo X API is not running. Please start the API first with run_app.bat
    echo Exiting...
    exit /b 1
) else (
    echo ✓ API is running
)

REM Check if monitoring stack is running
echo Checking if monitoring stack is running...
curl -s http://localhost:9090/-/healthy > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo X Prometheus is not running. Please start the monitoring stack with:
    echo docker-compose up -d
    echo Exiting...
    exit /b 1
) else (
    echo ✓ Monitoring stack is running
)

REM Run test scenarios
echo.
echo ===================================================================
echo                  Running Error Test Scenarios
echo ===================================================================

REM Scenario 1: High error rate
echo.
echo Scenario 1: Generating high error rate to trigger alert
echo Generating 30 error requests with 1 second delay between requests...

for /L %%i in (1,1,30) do (
    echo Sending error request %%i/30...
    curl -s -X POST -H "Content-Type: application/json" ^
        -d "{\"MinTemp\": \"invalid\", \"MaxTemp\": \"NaN\", \"invalidField\": true}" ^
        http://localhost:5050/predict > nul 2>&1
    timeout /t 1 > nul
)

echo.
echo Test complete!
echo You can now check AlertManager to see if alerts have been triggered:
echo http://localhost:9093
echo.
echo Check the Grafana dashboard to see the error rate spike:
echo http://localhost:3000/d/weather/weather-prediction-api-dashboard
