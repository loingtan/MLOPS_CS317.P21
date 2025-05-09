#!/bin/bash


DOCKER_USERNAME="maikusobu" 
REMOTE_SERVER="100.114.94.56"
REMOTE_USER="mlops"
APP_NAME="weather-prediction"
REMOTE_DIR="/home/mlops/mlops-lab1/Demo"


GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment process...${NC}"


echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${DOCKER_USERNAME}/${APP_NAME}:latest .


echo -e "${GREEN}Logging in to Docker Hub...${NC}"
docker login

echo -e "${GREEN}Pushing image to Docker Hub...${NC}"
docker push ${DOCKER_USERNAME}/${APP_NAME}:latest

echo -e "${GREEN}Creating .env file...${NC}"
echo "DOCKER_USERNAME=${DOCKER_USERNAME}" > .env
echo -e "${GREEN}Copying files to remote server...${NC}"
scp docker-compose.yml .env ${REMOTE_USER}@${REMOTE_SERVER}:${REMOTE_DIR}/
echo -e "${GREEN}Deploying on remote server...${NC}"
ssh ${REMOTE_USER}@${REMOTE_SERVER} "cd ${REMOTE_DIR} && \
    docker compose pull && \
    docker compose up -d"

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "Your application should now be running at http://${REMOTE_SERVER}:5000" 