@echo off
echo Starting the Weather Prediction API...

REM Set environment variables
set PYTHONPATH=.

REM Create directory for logs if it doesn't exist
if not exist "app\logs" mkdir app\logs

REM Install required packages if needed
echo Checking for required packages...
echo Installing required packages with exact versions specified in requirements.txt...
pip install --no-cache-dir -r app/requirements.txt

REM Force reinstall scikit-learn to ensure correct version
echo Ensuring scikit-learn 1.6.1 is installed (required for compatibility with pickled models)...
pip install --no-cache-dir --force-reinstall scikit-learn==1.6.1

REM Run FastAPI app with output redirection for logging
echo The application will be available at http://localhost:5050
echo The API documentation will be available at http://localhost:5050/docs
echo Press Ctrl+C to stop the application

cd app
python -m uvicorn app:app --host=0.0.0.0 --port=5050 --reload > stdout.log 2> stderr.log
