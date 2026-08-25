"""Score the CNN on held-out images and plot per-class accuracy.

Rebuilds ground truth from error_data.txt using the same 0.5 mm / 0.2 deg
thresholds as the dataset builders, then reports overall and per-class accuracy
and writes a bar chart.

This measures CLASSIFICATION accuracy, not insertion success. It cannot
reproduce Tables 4.1-4.3, which were counted by hand from the physical trials.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg19
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# User Configuration
# =========================================================

MODEL_PATH = "checkpoints/cnn_misalignment_classifier.pth"

TEST_IMAGE_DIR = "data/test_images"
ERROR_DATA_FILE = "data/test_error_data.txt"

MODEL_TYPE = "resnet"        # "resnet" or "vgg"
TRAINING_MODE = "train"      # "train" or "fine_tune"

IMAGE_SIZE = (336, 336)


# =========================================================
# Class Labels (27 total, must match training)
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


# =========================================================
# Movement classification thresholds
# =========================================================

DX_THRESHOLD = 0.0005
DY_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.2

def classify_label(dx, dy, dtheta):
    if dx > DX_THRESHOLD: mx = "Move Down"
    elif dx < -DX_THRESHOLD: mx = "Move Up"
    else: mx = "No Move"

    if dy > DY_THRESHOLD: my = "Move Right"
    elif dy < -DY_THRESHOLD: my = "Move Left"
    else: my = "No Move"

    if abs(dtheta) <= ROT_THRESHOLD:
        rot = "No Rotate"
    else:
        rot = "Rotate Clockwise" if dtheta > 0 else "Rotate Counterclockwise"

    return f"{mx}, {my}, {rot}"


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
            raise ValueError("model_type must be 'resnet' or 'vgg'")

        # Fully connected head consistent with training setup
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
# Testing Procedure
# =========================================================

def test_cnn():
    print(f"Testing CNN model: {MODEL_TYPE.upper()}  |  Mode: {TRAINING_MODE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MisalignmentClassifier(MODEL_TYPE, TRAINING_MODE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    ground_truth = {}
    with open(ERROR_DATA_FILE, "r") as f:
        for i, line in enumerate(f.readlines()):
            dx, dy, dtheta = map(float, line.strip().split())
            ground_truth[f"{i+1}.png"] = classify_label(dx, dy, dtheta)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(".png")])
    total_images = len(image_files)
    correct_predictions = 0

    per_class_total = {label: 0 for label in CLASS_LABELS}
    per_class_correct = {label: 0 for label in CLASS_LABELS}

    print(f"Processing {total_images} test images")

    for image_name in tqdm(image_files):
        true_label = ground_truth.get(image_name)
        if true_label is None:
            continue

        img_path = os.path.join(TEST_IMAGE_DIR, image_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_idx = torch.argmax(model(img)).item()
            pred_label = CLASS_LABELS[pred_idx]

        per_class_total[true_label] += 1
        if pred_label == true_label:
            correct_predictions += 1
            per_class_correct[true_label] += 1

        print(f"{image_name}")
        print(f"GT:   {true_label}")
        print(f"PRED: {pred_label}")
        print("----------------------------------")

    overall_acc = (correct_predictions / total_images) * 100
    print(f"Overall Test Accuracy: {overall_acc:.2f}%  ({correct_predictions}/{total_images})")

    per_class_acc = {
        label: (per_class_correct[label] / per_class_total[label] * 100)
        if per_class_total[label] > 0 else 0
        for label in CLASS_LABELS
    }

    plt.figure(figsize=(14, 7))
    plt.bar(CLASS_LABELS, per_class_acc.values(), color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Per-Class Accuracy - {MODEL_TYPE.upper()} ({TRAINING_MODE})")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("cnn_per_class_accuracy.png")
    plt.show()


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    test_cnn()

