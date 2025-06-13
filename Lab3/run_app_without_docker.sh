#!/bin/bash
# Startup script to run FastAPI app directly on host and monitoring stack in Docker
# filepath: c:\Users\09398\Subject\Mlops\MLOPS_CS317.P21\Lab3\run_app_without_docker.sh

# Check if the required Python packages are installed
echo -e "\033[1;36mChecking required Python packages...\033[0m"
cd app

if [ ! -f "requirements.txt" ]; then
    echo -e "\033[1;31mError: requirements.txt not found in app directory\033[0m"
    exit 1
fi

echo -e "\033[1;36mInstalling required Python packages...\033[0m"
pip install --no-cache-dir -r requirements.txt

# Force reinstall scikit-learn to ensure correct version
echo -e "\033[1;36mEnsuring scikit-learn 1.6.1 is installed (required for compatibility with pickled models)...\033[0m"
pip install --no-cache-dir --force-reinstall scikit-learn==1.6.1

# Start monitoring stack (Prometheus, Grafana, etc.) in Docker
echo -e "\033[1;36mStarting monitoring stack with Docker Compose...\033[0m"
cd ..
docker-compose up -d prometheus grafana alertmanager node-exporter

# Wait for services to start
echo -e "\033[1;36mWaiting for monitoring services to start...\033[0m"
sleep 5

# Print access information
echo -e "\n\033[1;32mMonitoring stack is now running!\033[0m"
echo -e "\033[1;33mAccess the following services in your browser:\033[0m"
echo -e "\033[1;33m  - Prometheus: http://localhost:9090\033[0m"
echo -e "\033[1;33m  - Grafana: http://localhost:3000 (default credentials: admin/admin)\033[0m"

# Start FastAPI application locally
echo -e "\n\033[1;36mStarting FastAPI application locally...\033[0m"
echo -e "\033[1;33mThe API will be available at http://localhost:5050\033[0m"
echo -e "\033[1;33mAPI documentation will be available at http://localhost:5050/docs\033[0m"
echo -e "\033[1;33mPress Ctrl+C to stop the application\n\033[0m"

cd app
python -m uvicorn app:app --host 0.0.0.0 --port 5050 --reload
