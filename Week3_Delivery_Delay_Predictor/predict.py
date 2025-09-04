import torch
from model import DelayPredictor
from preprocess import encode_features

model = DelayPredictor(input_dim=5)
model.load_state_dict(torch.load("model_weights.pth"))  # Load trained weights
model.eval()

def predict_delay(input_data):
    features = torch.tensor(encode_features(input_data)).unsqueeze(0).float()
    with torch.no_grad():
        delay = model(features).item()
    return round(delay, 2)
