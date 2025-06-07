#!/bin/bash
# Error simulation script for testing the alerting functionality

echo "==================================================================="
echo "         Error Simulation for Alerting System Testing"
echo "==================================================================="
echo ""

# Function to generate error requests
generate_errors() {
    local num_errors=$1
    local delay=$2
    
    echo "Generating $num_errors error requests with $delay second delay between requests..."
    
    for i in $(seq 1 $num_errors); do
        # Send an invalid request to trigger an error
        echo -n "Sending error request $i/$num_errors... "
        
        # Invalid JSON request that will cause an error
        curl -s -X POST -H "Content-Type: application/json" \
            -d '{"MinTemp": "invalid", "MaxTemp": "NaN", "invalidField": true}' \
            http://localhost:5000/predict > /dev/null
            
        echo "Done."
        # Wait for specified delay
        sleep $delay
    done
}

# Check if API is running
if ! curl -s http://localhost:5000/health > /dev/null; then
    echo "❌ API is not running. Please start the API first with ./run_app.sh"
    exit 1
fi

echo "This script will generate error requests to trigger the alerting system."
echo "It will help demonstrate how the monitoring system detects and alerts on high error rates."
echo ""
echo "The script will send 20 invalid requests to the API, causing a high error rate."
echo ""
read -p "Press Enter to start generating errors or Ctrl+C to cancel..."

# Generate a series of error requests
generate_errors 20 0.5

echo ""
echo "==================================================================="
echo "Error simulation completed!"
echo "==================================================================="
echo ""
echo "Check the following to observe the results:"
echo "1. Prometheus alert status: http://localhost:9090/alerts"
echo "2. AlertManager: http://localhost:9093"
echo "3. Error rate graph in Grafana: http://localhost:3000"
echo ""
echo "Note: It may take a minute or two for the alerts to trigger based on the"
echo "configured thresholds and evaluation intervals."
echo "==================================================================="
