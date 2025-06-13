@echo off
REM Update script for Loki integration
REM This script updates all necessary components for the Loki centralized logging

echo Starting Loki integration update...

REM Create Loki directory if it doesn't exist
if not exist .\loki mkdir .\loki

REM Check if loki-config.yaml exists, create if not
if not exist .\loki\loki-config.yaml (
    echo Creating Loki configuration file...
    > .\loki\loki-config.yaml (
        echo auth_enabled: false
        echo.
        echo server:
        echo   http_listen_port: 3100
        echo   grpc_listen_port: 9096
        echo.
        echo common:
        echo   path_prefix: /loki
        echo   storage:
        echo     filesystem:
        echo       chunks_directory: /loki/chunks
        echo       rules_directory: /loki/rules
        echo   replication_factor: 1
        echo   ring:
        echo     instance_addr: 127.0.0.1
        echo     kvstore:
        echo       store: inmemory
        echo.
        echo schema_config:
        echo   configs:
        echo     - from: 2020-10-24
        echo       store: boltdb-shipper
        echo       object_store: filesystem
        echo       schema: v11
        echo       index:
        echo         prefix: index_
        echo         period: 24h
        echo.
        echo ruler:
        echo   alertmanager_url: http://alertmanager:9093
        echo.
        echo limits_config:
        echo   retention_period: 7d
        echo   enforce_metric_name: false
        echo   reject_old_samples: true
        echo   reject_old_samples_max_age: 168h
        echo   max_entries_limit_per_query: 5000
        echo.
        echo compactor:
        echo   working_directory: /loki/boltdb-shipper-compactor
        echo   shared_store: filesystem
        echo   compaction_interval: 10m
        echo   retention_enabled: true
        echo   retention_delete_delay: 2h
        echo   retention_delete_worker_count: 150
        echo.
        echo analytics:
        echo   reporting_enabled: false
    )
    echo Loki configuration created.
) else (
    echo Loki configuration already exists.
)

REM Restart all services to apply changes
echo Restarting services to apply changes...
docker-compose down
docker-compose up -d

echo Waiting for services to start...
timeout /t 10 /nobreak > NUL

REM Check if services are running
echo Checking service status...
docker-compose ps

echo Loki integration complete!
echo You can now access logs via Grafana at http://localhost:3000
echo Login with admin/admin and navigate to Explore ^> Select Loki as datasource
