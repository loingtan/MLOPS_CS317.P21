FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

# Use single worker process in production with Uvicorn
# Multiple workers would require additional configuration for Prometheus metrics
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050", "--workers", "1"]
