#!/bin/bash
# Cleanup script for the monitoring stack and application

echo "==================================================================="
echo "           Cleanup for Weather Prediction API Monitoring"
echo "==================================================================="
echo ""

# Function to stop the Flask app if it's running
stop_flask_app() {
    echo "Stopping Flask application if running..."
    pkill -f "python -m flask run" || echo "No Flask application running."
}

# Function to stop and remove Docker containers
stop_docker_containers() {
    echo "Stopping and removing Docker containers..."
    docker-compose down

    # Force remove containers that might be leftover
    echo "Checking for leftover containers..."
    CONTAINERS=("prometheus" "node-exporter" "alertmanager" "grafana" "fluent-bit")
    
    for container in "${CONTAINERS[@]}"; do
        if [ "$(docker ps -a -q -f name=^/${container}$)" ]; then
            echo "Removing container: ${container}"
            docker rm -f ${container} 2>/dev/null
        fi
    done
    
    # Check if ports are still bound and try to kill processes
    for port in 9090 9100 9093 3000 2020; do
        if lsof -i:${port} >/dev/null 2>&1; then
            echo "Port ${port} is in use. Attempting to free it..."
            
            # Get PID of process using the port
            PID=$(lsof -t -i:${port} 2>/dev/null)
            
            if [ ! -z "$PID" ]; then
                echo "Killing process $PID using port ${port}..."
                kill -9 $PID 2>/dev/null || sudo kill -9 $PID 2>/dev/null || echo "Failed to kill process. You may need sudo privileges."
            fi
        fi
    done
}

# Function to remove log files
clean_logs() {
    echo "Removing log files..."
    rm -f app/*.log
    echo "Log files removed."
}

# Check for --force flag
if [[ "$1" != "--force" ]]; then
    # Ask for confirmation if not forced
    echo "This script will:"
    echo "1. Stop the Flask application (if running)"
    echo "2. Stop and remove all Docker containers for the monitoring stack"
    echo "3. Remove log files"
    echo ""
    echo "Data in Docker volumes will be preserved unless you specify --volumes."
    echo ""
    read -p "Do you want to proceed? (y/n): " confirm

    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Operation cancelled."
        exit 0
    fi
else
    echo "Forced cleanup - skipping confirmation."
fi

# Check if volumes should be removed
if [[ "$1" == "--volumes" ]]; then
    echo "WARNING: You've chosen to remove Docker volumes as well."
    echo "This will delete all collected metrics and dashboard configurations."
    read -p "Are you sure? (y/n): " confirm_volumes
    
    if [[ "$confirm_volumes" == "y" || "$confirm_volumes" == "Y" ]]; then
        docker-compose down -v
        echo "Docker containers and volumes removed."
    else
        stop_docker_containers
    fi
else
    stop_docker_containers
fi

# Stop the Flask app and clean logs
stop_flask_app
clean_logs

echo ""
echo "==================================================================="
echo "Cleanup completed successfully!"
echo "==================================================================="
echo ""
echo "To restart the system:"
echo "1. Run ./setup.sh to set up the environment"
echo "2. Start Docker containers: docker-compose up -d"
echo "3. Start the Flask app: ./run_app.sh"
echo "==================================================================="
