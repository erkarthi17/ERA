import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generate explanation for the delay
def generate_explanation(input_data, delay):
    prompt = f"""
Delivery from {input_data['origin']} to {input_data['destination']} is delayed by {delay} minutes.
Traffic level: {input_data['traffic_level']}, Weather: {input_data['weather']}, Vehicle: {input_data['vehicle_type']}.
Explain the delay in natural language for a customer update. Keep in mind that the customer is not a technical person and you need to explain the delay in a way that is easy to understand.
And, if the delay is negative, then greet the customer with a positive message. Else, explain the delay in a way that is easy to understand.
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
