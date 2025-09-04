from fastapi import FastAPI
from predict import predict_delay
from gemini import generate_explanation
import uvicorn

# Create the FastAPI app
app = FastAPI()

@app.post("/predict")
def get_prediction(input_data: dict):
    delay = predict_delay(input_data)
    explanation = generate_explanation(input_data, delay)
    return {
        "predicted_delay_minutes": delay,
        "explanation": explanation
    }

# Run the app

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)
