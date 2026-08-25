import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg19
from flask import Flask, request, jsonify
from PIL import Image
import tempfile

# =========================================================
# User Configurable Parameters
# =========================================================

MODEL_PATH = "checkpoints/cnn_misalignment_classifier.pth"
MODEL_TYPE = "resnet"          # "resnet" or "vgg"
TRAINING_MODE = "train"        # "train" or "fine_tune"
IMAGE_SIZE = (336, 336)
SERVER_PORT = 4003


# =========================================================
# Class Labels (must match training)
# =========================================================

movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

CLASS_LABELS = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]

IDX_TO_CLASS = {idx: label for idx, label in enumerate(CLASS_LABELS)}


# =========================================================
# Model Definition (must match training architecture)
# =========================================================

class MisalignmentClassifier(nn.Module):
    def __init__(self, model_type="resnet", training_mode="train"):
        super().__init__()

        if model_type == "resnet":
            self.backbone = resnet18(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_type == "vgg":
            self.backbone = vgg19(pretrained=False)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError("Unsupported model type.")

        # Head structure consistent with training script
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, len(CLASS_LABELS))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =========================================================
# Load Model
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MisalignmentClassifier(model_type=MODEL_TYPE, training_mode=TRAINING_MODE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()


# =========================================================
# Image Preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


# =========================================================
# Flask API
# =========================================================

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image_file = request.files["image"]

    # Save to temp file for PIL-safe loading
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        os.remove(image_path)
        return jsonify({"error": f"Image read error: {e}"}), 500
    finally:
        os.remove(image_path)

    with torch.no_grad():
        outputs = model(img)
        pred_idx = torch.argmax(outputs).item()
        pred_label = IDX_TO_CLASS[pred_idx]

    return jsonify({
        "predicted_label": pred_label,
        "index": pred_idx
    })


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":
    print(f"Starting CNN inference server on port {SERVER_PORT}")
    print(f"Model: {MODEL_TYPE.upper()} | Mode: {TRAINING_MODE} | GPU: {torch.cuda.is_available()}")
    app.run(host="0.0.0.0", port=SERVER_PORT)
