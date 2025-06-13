# Stop script for monitoring stack
# filepath: c:\Users\09398\Subject\Mlops\MLOPS_CS317.P21\Lab3\stop_without_docker.ps1

Write-Host "Stopping monitoring stack..." -ForegroundColor Cyan
docker-compose down

Write-Host "`nAll services have been stopped." -ForegroundColor Green
Write-Host "Note: The FastAPI application needs to be stopped manually if it's running in another terminal." -ForegroundColor Yellow
