# Delivery Delay Predictor

This project implements a machine learning-powered API to predict potential delivery delays and provide natural language explanations for those predictions. It uses a PyTorch neural network for the core prediction logic and integrates with the Google Gemini API to generate user-friendly explanations. The API is built with FastAPI, making it easy to integrate into other applications.

## Features

*   **Delay Prediction:** Predicts delivery delays in minutes based on various factors like distance, traffic, weather, vehicle type, and time of day.
*   **AI-Powered Explanations:** Utilizes the Google Gemini API to generate clear, concise explanations for predicted delays.
*   **FastAPI Backend:** Provides a robust and scalable API endpoint for predictions.
*   **PyTorch Model:** Employs a simple feed-forward neural network for delay estimation.

## Project Structure

├── .env # Environment variables (e.g., GEMINI_API_KEY) - IGNORED BY GIT
├── .gitignore # Specifies files/folders to ignore in Git
├── app.py # FastAPI application main entry point
├── gemini.py # Handles interaction with the Google Gemini API
├── model.py # Defines the PyTorch neural network architecture
├── model_weights.pth # Trained PyTorch model weights - IGNORED BY GIT
├── payload.json # Example input payload for testing the API
├── predict.py # Contains the logic for loading the model and making predictions
├── preprocess.py # Functions for encoding input features
├── README.md # This README file
├── requirements.txt # Python dependencies
├── test_client.py # Script to test the FastAPI /predict endpoint
└── train.py # Script to train the PyTorch delay prediction model


## Setup and Installation

Follow these steps to get the project up and running:

1.  **Clone the Repository (if not already done):**
    ```bash
    git clone <your-repository-url>
    cd deliver_delay_predictor
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Google Gemini API Key:**
    *   Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a file named `.env` in the `deliver_delay_predictor` directory.
    *   Add your API key to this file:
        ```
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
        ```
    *   **Important:** Ensure there are no spaces around the `=` sign and the key is enclosed in double quotes. This file is ignored by `.gitignore` for security.

## Usage

### 1. Train the Prediction Model (Optional, if `model_weights.pth` exists)

If you need to re-train the model or `model_weights.pth` does not exist, run the training script. This will generate a `model_weights.pth` file in the project directory.

```bash
python train.py
```

### 2. Start the FastAPI Application

Open a terminal, navigate to the `deliver_delay_predictor` directory, and run the FastAPI server:

```bash
python -m uvicorn app:app --reload --port 8002
```
You should see output indicating the server is running, typically at `http://127.0.0.1:8002`. Keep this terminal window open.

### 3. Test the Prediction Endpoint

You can test the API by sending a POST request using the provided `test_client.py` script.

#### Preparing the Payload:

Edit `payload.json` to define the input data for your prediction:

```json
{
  "distance_km": 120,
  "traffic_level": "High",
  "weather": "Rain",
  "vehicle_type": "Truck",
  "time_of_day": "Afternoon",
  "origin": "Castle Donington",
  "destination": "London"
}
```

#### Running the Client:

Open a **new terminal window**, navigate to the `deliver_delay_predictor` directory, and run the client script:

```bash
python test_client.py
```

This will print the HTTP status code and the JSON response from the API, which includes the predicted delay and a Gemini-generated explanation.

## API Endpoint

### `POST /predict`

*   **Description:** Predicts the delivery delay based on various input features and provides a natural language explanation.
*   **Request Body (JSON):**
    ```json
    {
      "distance_km": <integer>,
      "traffic_level": "Low" | "Medium" | "High",
      "weather": "Clear" | "Rain" | "Snow" | "Fog",
      "vehicle_type": "Van" | "Truck" | "Bike",
      "time_of_day": "Morning" | "Afternoon" | "Evening" | "Night",
      "origin": <string>,         # Used for explanation
      "destination": <string>     # Used for explanation
    }
    ```
*   **Response Body (JSON):**
    ```json
    {
      "predicted_delay_minutes": <float>,
      "explanation": <string>
    }
    ```

## Model Details

The project uses a simple feed-forward neural network implemented in PyTorch (`model.py`). This model is trained on simulated data (`train.py`) to learn the relationship between input features and delivery delays. The trained weights are saved in `model_weights.pth`.

## Gemini API Integration

The Google Gemini API (`gemini.py`) is used to take the predicted delay and the original input features (including `origin` and `destination`) to craft a user-friendly, descriptive explanation for the delivery status.
