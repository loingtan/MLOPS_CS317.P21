#!/bin/bash
# Stop script for monitoring stack
# filepath: c:\Users\09398\Subject\Mlops\MLOPS_CS317.P21\Lab3\stop_without_docker.sh

echo -e "\033[1;36mStopping monitoring stack...\033[0m"
docker-compose down

echo -e "\n\033[1;32mAll services have been stopped.\033[0m"
echo -e "\033[1;33mNote: The FastAPI application needs to be stopped manually if it's running in another terminal.\033[0m"
