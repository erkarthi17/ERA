import requests
import json

# FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Load payload from JSON file
with open('payload.json', 'r') as f:
    payload = json.load(f)

# Send POST request
response = requests.post(url, json=payload)

# Print result
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
