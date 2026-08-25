"""Serve the 27-class CNN alignment classifier over HTTP.

POST an image to /predict and get back one of the 27 correction strings on the
key "movement". This is the endpoint complete_system/align_and_insert_cnn.py
drives its control loop from.

MODEL_TYPE selects the backbone; VGG-19 is the one selected for the
end-to-end evaluations. The architecture here must match train_cnn_27class.py
exactly or the state dict will not load.

Defaults to port 4001. Analysis only; does not touch the robot.
"""

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

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/cnn_vgg19_27class.pth")

# "vgg" (the selected backbone) or "resnet". Leave unset to detect the
# architecture from the checkpoint itself - the archive holds both, and loading
# a ResNet-18 state dict into a VGG-19 (or the reverse) fails with a wall of
# unexpected-key errors that says nothing about the real cause.
MODEL_TYPE = os.environ.get("MODEL_TYPE")
TRAINING_MODE = "train"        # "train" or "fine_tune"
IMAGE_SIZE = (336, 336)

# 4001 is the port align_and_insert_cnn.py posts to; 4002 is the port
# vira/align_single_stage_cnn.py posts to. Override with PORT.
SERVER_PORT = int(os.environ.get("PORT", 4001))


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
    def __init__(self, model_type="resnet", training_mode="train", hidden=None):
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

        # Two head shapes exist in the archive and they are NOT weight-compatible:
        # deep=False  nf -> 1024 -> 27          (the simpler variant)
        # deep=True   nf -> 1024 -> 512 -> 27   (as the deployed VGG-19 was built)
        # The deployed checkpoint behind the 90.3% / 83.1% results is the deep one.
        if deep_head:
            self.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, len(CLASS_LABELS)),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, len(CLASS_LABELS)),
            )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =========================================================
# Load Model
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(MODEL_PATH, map_location=device)

if MODEL_TYPE is None:
    # VGG-19 keeps its conv stack under "features", ResNet-18 under "layerN".
    keys = state_dict.keys()
    if any(k.startswith("backbone.features") for k in keys):
        MODEL_TYPE = "vgg"
    elif any(k.startswith("backbone.layer") for k in keys):
        MODEL_TYPE = "resnet"
    else:
        raise SystemExit(
            f"Cannot tell which backbone {MODEL_PATH} holds. "
            "Set MODEL_TYPE=vgg or MODEL_TYPE=resnet explicitly."
        )
    print(f"Detected backbone from checkpoint: {MODEL_TYPE}")

# A third Linear layer means a hidden layer between 1024 and the 27 classes.
# Read its width from the weight rather than guessing: the archive holds both
# 512 and 1024 variants.
HIDDEN = None
for key in ("fc.2.weight", "classifier.2.weight"):
    w = state_dict.get(key)
    if w is not None and w.shape[0] != len(CLASS_LABELS):
        HIDDEN = w.shape[0]
        break
print(f"Detected head: 1024 -> {f'{HIDDEN} -> ' if HIDDEN else ''}{len(CLASS_LABELS)}")

model = MisalignmentClassifier(
    model_type=MODEL_TYPE, training_mode=TRAINING_MODE, hidden=HIDDEN
)

# The two archived trainers also disagree on the attribute name for the head.
if any(k.startswith("classifier.") for k in state_dict):
    state_dict = {
        (k.replace("classifier.", "fc.", 1) if k.startswith("classifier.") else k): v
        for k, v in state_dict.items()
    }

model.load_state_dict(state_dict)
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

    # "movement" is the key every robot client in this repo reads.
    # "predicted_label" is kept as an alias for older callers.
    return jsonify({
        "movement": pred_label,
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
