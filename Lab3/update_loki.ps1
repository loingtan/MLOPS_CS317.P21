# Update script for Loki integration
# This script updates all necessary components for the Loki centralized logging

Write-Host "Starting Loki integration update..." -ForegroundColor Cyan

# Create Loki directory if it doesn't exist
if (-not (Test-Path -Path ".\loki")) {
    Write-Host "Creating Loki directory..." -ForegroundColor Yellow
    New-Item -Path ".\loki" -ItemType Directory | Out-Null
}

# Check if loki-config.yaml exists, create if not
if (-not (Test-Path -Path ".\loki\loki-config.yaml")) {
    Write-Host "Creating Loki configuration file..." -ForegroundColor Yellow
    $lokiConfig = @"
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://alertmanager:9093

limits_config:
  retention_period: 7d
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  max_entries_limit_per_query: 5000
  
compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150

analytics:
  reporting_enabled: false
"@

    Set-Content -Path ".\loki\loki-config.yaml" -Value $lokiConfig
    Write-Host "Loki configuration created." -ForegroundColor Green
}
else {
    Write-Host "Loki configuration already exists." -ForegroundColor Green
}

# Restart all services to apply changes
Write-Host "Restarting services to apply changes..." -ForegroundColor Yellow
docker compose down
docker compose up -d

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if services are running
Write-Host "Checking service status..." -ForegroundColor Yellow
docker compose ps

Write-Host "Loki integration complete!" -ForegroundColor Green
Write-Host "You can now access logs via Grafana at http://localhost:3000" -ForegroundColor Cyan
Write-Host "Login with admin/admin and navigate to Explore > Select Loki as datasource" -ForegroundColor Cyan
