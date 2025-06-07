#!/bin/bash
# Test script for the Weather Prediction API Monitoring Demo

echo "==================================================================="
echo "         Weather Prediction API Monitoring & Logging Test"
echo "==================================================================="
echo ""

# Check if the API is running
echo "Checking if the API is running..."
if ! curl -s http://localhost:5000/health > /dev/null; then
    echo "❌ API is not running. Please start the API first with ./run_app.sh"
    echo "Exiting..."
    exit 1
else
    echo "✅ API is running."
fi

# Check if monitoring stack is running
echo "Checking if monitoring stack is running..."
if ! curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "❌ Prometheus is not running. Please start the monitoring stack with:"
    echo "docker-compose up -d"
    echo "Exiting..."
    exit 1
else
    echo "✅ Prometheus is running."
fi

echo ""
echo "==================================================================="
echo "Starting test scenarios..."
echo "==================================================================="
echo ""

# Run the test_api.py script
echo "Running API tests..."
python3 test_api.py

echo ""
echo "==================================================================="
echo "Test complete! You can now check the monitoring dashboards:"
echo "==================================================================="
echo ""
echo "📊 Grafana: http://localhost:3000 (login: admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
echo "🚨 AlertManager: http://localhost:9093"
echo "📝 Fluent Bit: http://localhost:2020"
echo ""
echo "To see the complete metrics from the API, visit:"
echo "http://localhost:5000/metrics"
echo ""
echo "==================================================================="
