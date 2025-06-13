# Create required directories
New-Item -ItemType Directory -Force -Path app\preprocessing | Out-Null
New-Item -ItemType Directory -Force -Path app\logs | Out-Null

# Check if Docker is running
try {
    docker info | Out-Null
}
catch {
    Write-Host "Docker is not running! Please start Docker and try again."
    exit 1
}

# Copy model files from Lab 2
Write-Host "Copying model files from Lab 2..."
if (Test-Path -Path "..\Lab2\model.pkl") {
    Copy-Item -Path "..\Lab2\model.pkl" -Destination "app\" -Force
    Write-Host "✓ Model copied successfully"
}
else {
    Write-Host "❌ Warning: model.pkl not found in Lab 2"
    Write-Host "Please make sure you have completed Lab 2 before continuing."
    $continue_choice = Read-Host "Continue anyway? (y/n)"
    if ($continue_choice -ne "y" -and $continue_choice -ne "Y") {
        exit 1
    }
}

if (Test-Path -Path "..\Lab2\preprocessing\model.pkl") {
    Copy-Item -Path "..\Lab2\preprocessing\model.pkl" -Destination "app\preprocessing\" -Force
    Write-Host "✓ Preprocessor copied successfully"
}
else {
    Write-Host "❌ Warning: preprocessing model not found in Lab 2"
}

# Install Python requirements if needed
if (Test-Path -Path "app\requirements.txt") {
    Write-Host "Installing Python dependencies..."
    try {
        pip install -r app\requirements.txt
    }
    catch {
        Write-Host "❌ Warning: Failed to install some dependencies"
    }
}

# Verify Docker Compose exists
try {
    docker-compose --version | Out-Null
}
catch {
    Write-Host "❌ docker-compose could not be found. Please install Docker Compose."
    exit 1
}

# Ask to start monitoring stack
$choice = Read-Host "Start the monitoring stack now? (y/n)"

if ($choice -eq "y" -or $choice -eq "Y") {
    # Check if required ports are free before starting
    Write-Host "Checking if required ports are free..."
    $portsBusy = $false
    
    $ports = @(9090, 9100, 9093, 3000, 2020)
    foreach ($port in $ports) {
        $portInUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($portInUse) {
            Write-Host "Warning: Port $port is already in use!"
            $portsBusy = $true
        }
    }
    
    if ($portsBusy) {
        Write-Host "Some ports are already in use. Running cleanup..."
        if (Test-Path -Path ".\cleanup.sh") {
            bash ./cleanup.sh --force
        }
    }
    
    # Start the docker-compose stack
    Write-Host "Starting monitoring stack..."
    docker-compose up -d

    Write-Host "Monitoring services starting:"
    Write-Host "- Prometheus: http://localhost:9090"
    Write-Host "- Grafana: http://localhost:3000 (admin/admin)"
    Write-Host "- AlertManager: http://localhost:9093"
    Write-Host "- Fluent Bit: Collecting logs on port 2020"

    # Wait for services to start
    Write-Host "Waiting for services to start..."
    Start-Sleep -Seconds 10

    # Check if services are up
    Write-Host "Checking if services are running..."
    docker-compose ps
}

Write-Host ""
Write-Host "Setup complete!"
Write-Host ""
Write-Host "To run the FastAPI application, use:"
Write-Host "    cd app && uvicorn app:app --host=0.0.0.0 --port=5050 --reload"
Write-Host "    or simply: .\run_app.ps1"
Write-Host ""
Write-Host "The Weather Prediction API will be available at http://localhost:5050"
Write-Host "The API documentation will be available at http://localhost:5050/docs"
Write-Host "The metrics endpoint will be available at http://localhost:5050/metrics"
Write-Host ""
Write-Host "For a full monitoring experience:"
Write-Host "1. Start the monitoring stack: docker-compose up -d"
Write-Host "2. Run the FastAPI app: cd app && uvicorn app:app --host=0.0.0.0 --port=5050"
Write-Host "3. Open Grafana at http://localhost:3000 and log in with admin/admin"
Write-Host "4. View the 'Weather Prediction API Dashboard' for real-time metrics"