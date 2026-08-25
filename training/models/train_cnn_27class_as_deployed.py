"""Train the 27-class CNN exactly as the deployed model was trained.

THIS is the script that produced `cnn_camera1_crop2_aug_wrong_station123.pth`,
the VGG-19 behind the 90.3 % and 83.1 % results - the filename is hard-coded at
its `torch.save`.

It differs from train_cnn_27class.py in the classifier head, and the difference
matters because the two are not weight-compatible:

    as deployed (here):        25088 -> 1024 -> 512 -> 27   (three Linear layers)
    train_cnn_27class.py:      25088 -> 1024 -> 27          (two Linear layers)

The second is the simpler head: "a 1024-unit fully connected
layer followed by a softmax output over the 27 correction classes". The
deployed weights have the extra 512-unit layer. Loading one into the other
fails.

Use this script to reproduce the deployed model. Use train_cnn_27class.py if you
want the architecture as described it.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, vgg19
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

# Define Paths
json_file_path = os.path.join(BASE_DIR, "cnn_camera1_crop2_station123_wrong.json")

# Generate all 27 class labels in the correct order
movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

class_labels = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]

# Create label-to-index mapping
class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

# Dataset Class with Augmentations
class MovementDataset(Dataset):
    def __init__(self, json_file, transform=None, augment=False):
        self.transform = transform
        self.augment = augment
        self.image_info = []

        # Load data from JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        for entry in data:
            image_path = entry["image"]
            label_text = entry["label"]

            # Ensure labels are formatted consistently
            label_parts = label_text.split(", ")
            if len(label_parts) == 3:
                label_text = ", ".join(label_parts)

            # Map text label to class index
            if label_text in class_to_idx:
                label = class_to_idx[label_text]
                self.image_info.append((image_path, label))
            else:
                print(f"Warning: Label '{label_text}' not found in predefined classes.")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, label = self.image_info[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.augment:
            image = train_augment(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

train_augment = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.3),  # 30% chance to go grayscale
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor,
])

# Standard Transformations for Validation & Test
standard_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor,
])

# Load dataset
full_dataset = MovementDataset(json_file_path, transform=standard_transform, augment=True)

# Split into Train, Validation, and Test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

# Modified Model for 27-Class Classification
class ClassificationModel(nn.Module):
    def __init__(self, model_type='resnet', pretrained=True):
        super(ClassificationModel, self).__init__

        if model_type == 'resnet':
            self.backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity
        elif model_type == 'vgg':
            self.backbone = vgg19(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity
        else:
            raise ValueError("model_type must be 'resnet' or 'vgg'")

        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU,
            nn.Linear(1024, 512), # 27-class output
            nn.ReLU,
            nn.Linear(512, len(class_labels))  # 27-class output
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

# Train Model
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0001, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward
            optimizer.step
            running_loss += loss.item * images.size(0)

        val_loss = 0.0
        model.eval
        with torch.no_grad:
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item * images.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {running_loss/len(train_dataset):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict, "cnn_camera1_crop2_aug_wrong_station123.pth")

        scheduler.step

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = ClassificationModel(model_type="vgg", pretrained=True)

# Train
train_model(model, train_loader, val_loader, num_epochs=20, device=device)

# Evaluate on Test Set
def test_model(model, test_loader, device="cuda"):
    model = model.to(device)
    model.eval
    correct = 0
    total = 0

    with torch.no_grad:
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum.item
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

# Load Best Model & Evaluate
model.load_state_dict(torch.load("cnn_camera1_crop2_aug_wrong_station123.pth"))
test_model(model, test_loader)
