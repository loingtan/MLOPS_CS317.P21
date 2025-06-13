#!/bin/bash

# Update script for Loki integration
# This script updates all necessary components for the Loki centralized logging

echo "Starting Loki integration update..."

# Create Loki directory if it doesn't exist
mkdir -p ./loki

# Check if loki-config.yaml exists, create if not
if [ ! -f "./loki/loki-config.yaml" ]; then
    echo "Creating Loki configuration file..."
    cat > ./loki/loki-config.yaml << 'EOL'
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
EOL
    echo "Loki configuration created."
else
    echo "Loki configuration already exists."
fi

# Restart all services to apply changes
echo "Restarting services to apply changes..."
docker-compose down
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo "Loki integration complete!"
echo "You can now access logs via Grafana at http://localhost:3000"
echo "Login with admin/admin and navigate to Explore > Select Loki as datasource"
