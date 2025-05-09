FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn


COPY . .


EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_ENV=production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"] 