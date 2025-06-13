@echo off
REM Test script for the Weather Prediction API Monitoring Demo (Windows batch version)

echo ===================================================================
echo          Weather Prediction API Monitoring & Logging Test
echo ===================================================================
echo.

REM Check if the API is running
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
    echo ✓ Prometheus is running
)

echo.
echo ===================================================================
echo                       Running API Tests
echo ===================================================================

echo Running API tests...
python test_api.py

echo.
echo ===================================================================
echo Test complete! You can now check the monitoring dashboards:
echo ===================================================================
echo.
echo 📊 Grafana: http://localhost:3000 (login: admin/admin)
echo 🔍 Prometheus: http://localhost:9090
echo 🚨 AlertManager: http://localhost:9093
echo 📝 Fluent Bit: http://localhost:2020
echo.
echo To see the complete metrics from the API, visit:
echo http://localhost:5050/metrics
echo.
echo ==================================================================
