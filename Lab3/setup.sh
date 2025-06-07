#!/bin/bash
set -e

# Run cleanup first to ensure we don't have conflicts
if [ -f "./cleanup.sh" ]; then
    echo "Running cleanup to ensure no conflicts..."
    bash ./cleanup.sh --force >/dev/null 2>&1 || true
    echo "Cleanup completed."
fi

# Create required directories
mkdir -p app/preprocessing
mkdir -p app/logs

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running! Please start Docker and try again."
    exit 1
fi

# Copy model files from Lab 2
echo "Copying model files from Lab 2..."
if [ -f "../Lab2/model.pkl" ]; then
    cp ../Lab2/model.pkl app/ && echo "✓ Model copied successfully"
else
    echo "❌ Warning: model.pkl not found in Lab 2"
    echo "Please make sure you have completed Lab 2 before continuing."
    read -p "Continue anyway? (y/n): " continue_choice
    if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
        exit 1
    fi
fi

if [ -f "../Lab2/preprocessing/model.pkl" ]; then
    cp ../Lab2/preprocessing/model.pkl app/preprocessing/ && echo "✓ Preprocessor copied successfully"
else
    echo "❌ Warning: preprocessing model not found in Lab 2"
fi

# Install Python requirements if needed
if [ -f "app/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r app/requirements.txt || echo "❌ Warning: Failed to install some dependencies"
fi

# Verify Docker Compose exists
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose could not be found. Please install Docker Compose."
    exit 1
fi

# Ask to start monitoring stack
read -p "Start the monitoring stack now? (y/n): " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    # Check if required ports are free before starting
    echo "Checking if required ports are free..."
    PORTS_BUSY=false
    
    for port in 9090 9100 9093 3000 2020; do
        if lsof -i:${port} >/dev/null 2>&1; then
            echo "Warning: Port ${port} is already in use!"
            PORTS_BUSY=true
        fi
    done
    
    if [ "$PORTS_BUSY" = true ]; then
        echo "Some ports are already in use. Running cleanup to free them..."
        bash ./cleanup.sh --force
    fi
    
    # Start the docker-compose stack
    echo "Starting monitoring stack..."
    docker-compose up -d

    echo "Monitoring services starting:"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo "- AlertManager: http://localhost:9093"
    echo "- Fluent Bit: Collecting logs on port 2020"

    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 10

    # Check if services are up
    echo "Checking if services are running..."
    docker-compose ps
fi

echo ""
echo "Setup complete!"
echo ""
echo "To run the Flask application, use:"
echo "    python -m flask run --host=0.0.0.0 --port=5000"
echo ""
echo "The Weather Prediction API will be available at http://localhost:5000"
echo "The metrics endpoint will be available at http://localhost:5000/metrics"
echo ""
echo "For a full monitoring experience:"
echo "1. Start the monitoring stack: docker-compose up -d"
echo "2. Run the Flask app: python -m flask run --host=0.0.0.0 --port=5000"
echo "3. Open Grafana at http://localhost:3000 and log in with admin/admin"
echo "4. View the 'Weather Prediction API Dashboard' for real-time metrics"
