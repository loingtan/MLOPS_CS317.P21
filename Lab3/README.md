# Lab 3: Monitoring and Logging for Weather Prediction API

This lab adds comprehensive monitoring and logging capabilities to the Weather Prediction API built in Lab 2.

## Features

### Monitoring Service

- **System Resource Monitoring**:
  - CPU usage
  - RAM usage
  - Disk space & I/O
  - Network I/O (total transmitted, total received)

- **API Monitoring**:
  - Requests per second
  - Error rate
  - Latency (response time)

- **Model Monitoring**:
  - Inference speed (CPU time)
  - Confidence score

### Logging Service

- Captures logs from:
  - syslog: System logs to identify non-application issues
  - stdout: Console output from the application
  - stderr: Error output including tracebacks
  - Application log file: Custom app logs

### Alerting Service

- **Configurable alerts for:**
  - Error rate exceeds 50%
  - Confidence score falls below 0.6
  - High CPU usage (>80%)
  - High memory usage (>80%)
  - High disk usage (>85%)
  - Abnormal network I/O
  - High response latency

- **Notification options:**
  - Email notifications (configurable)
  - Support for Slack/Telegram notifications (commented examples provided)

## Architecture

- **Flask App**: Weather prediction service with integrated Prometheus metrics
- **Prometheus**: Time-series database for collecting and storing metrics
- **Grafana**: Visualization dashboards for real-time monitoring
- **AlertManager**: Handles alerts and notification routing
- **Fluent Bit**: Log collection and forwarding from multiple sources
- **Node Exporter**: Collects detailed system metrics

## Setup Instructions

### Automated Setup

Use the setup script to prepare the environment:

```bash
# Give execution permission
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The setup script will:
1. Create necessary directories
2. Copy model files from Lab 2
3. Install required dependencies
4. Start the monitoring stack (Docker Compose services)

### End-to-End Testing

For a complete test of the entire system, use the end-to-end test script:

```bash
# Give execution permission
chmod +x e2e_test.sh

# Run the end-to-end test
./e2e_test.sh
```

This script will:
1. Run cleanup to ensure a fresh start
2. Start the monitoring stack 
3. Verify all services are running properly
4. Start the Flask application
5. Test basic functionality
6. Verify metrics are being collected

### Manual Setup

1. Copy your model files from Lab 2:
```bash
mkdir -p app/preprocessing
cp ../Lab2/model.pkl app/
cp ../Lab2/preprocessing/model.pkl app/preprocessing/
```

2. Install Python dependencies:
```bash
pip install -r app/requirements.txt
```

3. Start the monitoring stack with Docker Compose:
```bash
docker-compose up -d
```

4. Run the Flask application:
```bash
# Using bash
./run_app.sh

# OR using PowerShell
./run_app.ps1

# OR using batch file on Windows
run_app.bat
```

## Accessing Services

- **Weather Prediction API**: http://localhost:5000
  - Endpoints: `/predict`, `/health`, `/metrics`
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (login: admin/admin)
- **AlertManager**: http://localhost:9093

## Testing the API

### Automated Test

Use the included test script to run a complete demonstration:

```bash
# After starting both the Flask app and monitoring stack
./test.sh
```

This script will:
1. Check if the API and monitoring stack are running
2. Run multiple test scenarios using `test_api.py`
3. Provide links to all dashboards at the end

### Manual Testing

You can also run the Python test script directly:

```bash
python test_api.py
```

This script will:
1. Send normal prediction requests
2. Generate deliberate errors to test error handling
3. Perform a load test with rapid requests
4. Check metrics after each test scenario

### Error Simulation

To test the alerting functionality:

```bash
./simulate_errors.sh
```

This will generate a series of error requests to trigger the error rate alert.

## Configuration Files

- **Prometheus**: 
  - Main config: `prometheus/prometheus.yml`
  - Alert rules: `prometheus/rules/alerts.rules`

- **AlertManager**: 
  - Config: `alertmanager/alertmanager.yml`
  - Email notifications need SMTP settings

- **Fluent Bit**:
  - Main config: `fluent-bit/fluent-bit.conf`
  - Log parsers: `fluent-bit/parsers.conf`

- **Grafana**:
  - Dashboard: `grafana/dashboard.json`
  - Data source: `grafana/datasource.yml`

## Customizing Alerts

Edit the `prometheus/rules/alerts.rules` file to modify alert thresholds and conditions. Alert routing and notifications can be configured in `alertmanager/alertmanager.yml`.

## Cleanup and Reset

To stop all services and clean up the environment:

```bash
./cleanup.sh
```

To remove Docker volumes (clearing all metrics history):

```bash
./cleanup.sh --volumes
```

## Extending the System

- Add GPU monitoring by installing NVIDIA drivers and using NVIDIA Docker runtime
- Integrate with additional notification channels like Slack or PagerDuty
- Add application-specific metrics for business KPIs
- Implement log rotation for handling high-volume logs
