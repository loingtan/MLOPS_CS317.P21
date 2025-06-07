@echo off
echo Starting the Weather Prediction API...

REM Set environment variables
set FLASK_APP=app/app.py
set FLASK_ENV=development
set PYTHONPATH=.

REM Create directory for logs if it doesn't exist
if not exist "app\logs" mkdir app\logs

REM Run Flask app with output redirection for logging
echo The application will be available at http://localhost:5000
echo Press Ctrl+C to stop the application

python -m flask run --host=0.0.0.0 --port=5000 > app\stdout.log 2> app\stderr.log
