#!/usr/bin/env python
"""
Test script for the Weather Prediction API
This sends a series of requests to test the API and generate metrics
"""

import requests
import json
import time
import random
from datetime import datetime

# API endpoint
BASE_URL = "http://localhost:5050"
PREDICT_URL = f"{BASE_URL}/predict"
HEALTH_URL = f"{BASE_URL}/health"
METRICS_URL = f"{BASE_URL}/metrics"

# Sample data for predictions
test_data = {
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Evaporation": 6.2,
    "Sunshine": 8.3,
    "WindGustDir": "W",
    "WindGustSpeed": 44.0,
    "WindDir9am": "W",
    "WindDir3pm": "WSW",
    "WindSpeed9am": 20.0,
    "WindSpeed3pm": 24.0,
    "Humidity9am": 71.0,
    "Humidity3pm": 22.0,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Cloud9am": 8.0,
    "Cloud3pm": 5.0,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No"
}


def make_random_data(error_prob=0.0):
    """Create randomized test data with optional deliberate errors"""
    data = test_data.copy()
    data["MinTemp"] = random.uniform(5, 20)
    data["MaxTemp"] = data["MinTemp"] + random.uniform(5, 15)
    data["Rainfall"] = random.uniform(0, 20)
    data["Humidity9am"] = random.uniform(40, 95)
    data["Humidity3pm"] = random.uniform(30, 90)
    data["RainToday"] = "Yes" if data["Rainfall"] > 1.0 else "No"

    # Add deliberate error conditions based on probability
    if random.random() < error_prob:
        error_type = random.choice(
            ["missing_field", "invalid_type", "extreme_value"])

        if error_type == "missing_field":
            # Remove a random field
            field_to_remove = random.choice(list(data.keys()))
            del data[field_to_remove]
            print(f"Introducing error: Removed field '{field_to_remove}'")

        elif error_type == "invalid_type":
            # Change a numeric field to string
            numeric_fields = ["MinTemp", "MaxTemp",
                              "Rainfall", "Humidity9am", "Humidity3pm"]
            field_to_change = random.choice(numeric_fields)
            data[field_to_change] = "invalid-value"
            print(
                f"Introducing error: Changed {field_to_change} to string value")

        elif error_type == "extreme_value":
            # Set an extreme value that might cause issues
            data["Humidity9am"] = 999.9
            print(f"Introducing error: Set extreme value for Humidity9am")

    return data


def check_health():
    """Check API health endpoint"""
    try:
        response = requests.get(HEALTH_URL)
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def make_prediction(data):
    """Make a prediction request"""
    try:
        start = time.time()
        response = requests.post(PREDICT_URL, json=data)
        duration = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print(
                f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}, Time: {duration:.4f}s")
        else:
            print(f"Error: {response.status_code} - {response.text}")

        return response
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def check_metrics():
    """Check metrics endpoint"""
    try:
        response = requests.get(METRICS_URL)
        if response.status_code == 200:
            # Print some key metrics
            lines = response.text.split('\n')
            metrics = [line for line in lines if not line.startswith(
                '#') and line.strip()]
            key_metrics = [m for m in metrics if any(
                pattern in m for pattern in ['model_confidence', 'http_requests_total', 'cpu_usage'])]

            print("\nSample metrics:")
            for m in key_metrics[:5]:  # Print first 5 key metrics
                print(f"  {m}")
            print(f"  ... and {len(metrics) - 5} more metrics")
        else:
            print(f"Error fetching metrics: {response.status_code}")
    except Exception as e:
        print(f"Metrics check failed: {e}")


def main():
    """Main test function"""
    print(f"Starting API test at {datetime.now()}")

    # Check if API is up
    if not check_health():
        print("API is not healthy, exiting.")
        return

    print("\n===== Test Scenario 1: Normal Operation =====")
    # Make a series of predictions with normal data
    num_requests = 15
    print(f"Making {num_requests} normal prediction requests...")

    for i in range(num_requests):
        data = make_random_data(error_prob=0.0)
        print(f"\nRequest {i+1}/{num_requests} (Normal)")
        make_prediction(data)
        time.sleep(0.5)

    # Check metrics after normal predictions
    print("\nChecking metrics after normal predictions:")
    check_metrics()

    print("\n===== Test Scenario 2: Error Conditions =====")
    # Make some requests with deliberate errors to test error handling
    num_error_requests = 5
    print(f"Making {num_error_requests} requests with deliberate errors...")

    for i in range(num_error_requests):
        data = make_random_data(error_prob=1.0)  # Force errors
        print(f"\nRequest {i+1}/{num_error_requests} (With Error)")
        make_prediction(data)
        time.sleep(0.5)

    # Check metrics after error predictions
    print("\nChecking metrics after error requests:")
    check_metrics()

    print("\n===== Test Scenario 3: Load Test =====")
    # Make rapid requests to simulate load
    num_load_requests = 10
    print(f"Making {num_load_requests} rapid requests...")

    for i in range(num_load_requests):
        data = make_random_data(error_prob=0.0)
        print(f"\rLoad request {i+1}/{num_load_requests}", end="")
        make_prediction(data)
        time.sleep(0.1)  # Reduced delay for load testing

    print("\n\nChecking metrics after load test:")
    check_metrics()

    print(f"\nTest completed at {datetime.now()}")


if __name__ == "__main__":
    main()
