import torch
import numpy as np
from model import DelayPredictor
from preprocess import encode_features

print("🚀 Training the model...")

# Simulated training samples
samples = [
    {"distance_km": 120, "traffic_level": "High", "weather": "Rain", "vehicle_type": "Truck", "time_of_day": "Afternoon", "delay": 45},
    {"distance_km": 80, "traffic_level": "Medium", "weather": "Clear", "vehicle_type": "Van", "time_of_day": "Morning", "delay": 15},
    {"distance_km": 200, "traffic_level": "High", "weather": "Snow", "vehicle_type": "Truck", "time_of_day": "Evening", "delay": 90},
    {"distance_km": 50, "traffic_level": "Low", "weather": "Clear", "vehicle_type": "Bike", "time_of_day": "Night", "delay": 5},
    {"distance_km": 150, "traffic_level": "Medium", "weather": "Fog", "vehicle_type": "Van", "time_of_day": "Afternoon", "delay": 40},
]

# Prepare training data
X = torch.tensor([encode_features(s) for s in samples], dtype=torch.float32)
y = torch.tensor([[s["delay"]] for s in samples], dtype=torch.float32)

# Initialize model
model = DelayPredictor(input_dim=5)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save trained weights
torch.save(model.state_dict(), "model_weights.pth")
print("✅ Training complete. Model weights saved to model_weights.pth")
