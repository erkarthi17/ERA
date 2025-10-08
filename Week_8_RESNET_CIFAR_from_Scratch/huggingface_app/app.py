import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr
from torchvision import transforms

# Allow importing model.py from the project root when running inside this subfolder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model import ResNet44V2  # uses the same architecture as training

# CIFAR-100 class names (fixed order used by torchvision)
CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
]

# Preprocessing identical to test-time transform used during training
transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Load model on CPU
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

def load_model():
    model = ResNet44V2(num_classes=100)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

MODEL = load_model()

def predict(image: Image.Image):
    # Convert to RGB, apply transforms
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    topk = torch.topk(probs, k=5)
    indices = topk.indices.tolist()
    scores = topk.values.tolist()

    return {CIFAR100_CLASSES[i]: float(s) for i, s in zip(indices, scores)}

TITLE = "CIFAR-100 ResNet (trained from scratch)"
DESC = "Upload or draw a 32×32-like image; the model will output top-5 CIFAR-100 class probabilities."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input image"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 predictions"),
    title=TITLE,
    description=DESC,
    examples=None,
)

if __name__ == "__main__":
    # Local dev: honor your localhost preference
    # On Spaces: standard launch is fine; Spaces manages host/port.
    if os.environ.get("SPACE"):
        demo.launch()
    else:
        demo.launch(server_name="localhost", server_port=7860)
