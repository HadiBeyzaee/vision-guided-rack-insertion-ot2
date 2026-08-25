"""Train the 27-class CNN alignment classifier.

VGG-19 or ResNet-18 backbone, a 1024-unit fully connected layer, and a softmax
over the 27 correction classes. Cross-entropy, Adam at 1e-4, StepLR every 5
epochs, 20 epochs at batch 16, images at 336x336.

VGG-19 is the variant used for the end-to-end evaluations; it
reached the highest insertion success at under one second per image-to-command
cycle.

KNOWN ISSUE, left as found: random_split returns Subset objects sharing one
underlying dataset, so rebinding val_ds.dataset.transform also rebinds it for
training. The augmentation configured at construction is therefore replaced
before the first epoch. See docs/caveats.md.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, vgg19, ResNet18_Weights, VGG19_Weights


# =========================================================
# User Configuration (Edit before running)
# =========================================================

# Path to CNN training dataset (converted from LLaVA JSON format)
DATASET_JSON = "data/cnn_training_dataset.json"

# Directory to store trained model checkpoints
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cnn_misalignment_classifier.pth")

# Model options
MODEL_TYPE = "resnet"      # Options: "resnet" or "vgg"
TRAINING_MODE = "train"    # Options: "train" or "fine_tune"

# Training configuration
IMAGE_SIZE = (336, 336)
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 8



# =========================================================
# Define class label set (27 classes)
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
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}


# =========================================================
# Dataset Loader
# =========================================================

class MisalignmentDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.samples = []

        with open(json_path, "r") as f:
            data = json.load(f)

        for entry in data:
            image_path = entry.get("image")
            label_text = entry.get("label")
            if label_text not in LABEL_TO_IDX:
                continue
            self.samples.append((image_path, LABEL_TO_IDX[label_text]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_idx


train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor,
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor,
])


# =========================================================
# Model Definition (27-class FC head)
# =========================================================

class MisalignmentClassifier(nn.Module):
    def __init__(self, model_type="resnet", pretrained=True, training_mode="train"):
        super.__init__

        self.training_mode = training_mode

        # Load pretrained CNN backbone
        if model_type == "resnet":
            self.backbone = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity

        elif model_type == "vgg":
            self.backbone = vgg19(
                weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity

        else:
            raise ValueError("model_type must be 'resnet' or 'vgg'")

        # Freeze backbone if fine-tuning
        if training_mode == "fine_tune":
            for param in self.backbone.parameters:
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU,
            nn.Linear(1024, len(CLASS_LABELS))
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return self.classifier(feats)


# =========================================================
# Training + Evaluation
# =========================================================

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward
            optimizer.step

            train_loss += loss.item * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss = 0.0
        model.eval
        with torch.no_grad:
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = criterion(model(imgs), labels)
                val_loss += loss.item * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict, CHECKPOINT_PATH)

        scheduler.step


def test_model(model, test_loader, device="cuda"):
    model = model.to(device)
    model.eval
    correct, total = 0, 0

    with torch.no_grad:
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = torch.max(model(imgs), 1)
            correct += (preds == labels).sum.item
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# =========================================================
# Execution
# =========================================================

if __name__ == "__main__":
    dataset = MisalignmentDataset(DATASET_JSON, transform=train_transform)

    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    val_ds.dataset.transform = val_test_transform
    test_ds.dataset.transform = val_test_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = MisalignmentClassifier(model_type=MODEL_TYPE, training_mode=TRAINING_MODE)

    print(f"Training Mode: {TRAINING_MODE}  |  Model: {MODEL_TYPE.upper}")
    train_model(model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    test_model(model, test_loader, device=device)
