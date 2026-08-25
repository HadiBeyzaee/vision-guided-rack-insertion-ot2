import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, vgg19
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A

# Define Paths
json_file_path = os.path.join(BASE_DIR, "cnn_augmented_camera1_crop3_merged_station5.json")

# Generate all 27 class labels in the correct order
movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]

class_labels = [
    f"{x}, {y}"
    for x in movement_x
    for y in movement_y

]

print(len(class_labels))

# Create label-to-index mapping
class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

class MovementDataset(Dataset):
    def __init__(self, json_file,  augment=True, augment_times=10, transform=None):
        self.augment = augment
        self.augment_times = augment_times
        self.transform = transform  # Add this line
        self.image_info = []

        # Load data from JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        for entry in data:
            image_path = entry["image"]
            label_text = entry["label"]

            # Normalize label format
            label_parts = label_text.split(", ")
            if len(label_parts) == 2:
                label_text = ", ".join(label_parts)

            # Add the sample multiple times if label is valid
            if label_text in class_to_idx:
                label = class_to_idx[label_text]
                for _ in range(self.augment_times):
                    self.image_info.append((image_path, label))
            else:
                print(f"Warning: Label '{label_text}' not found in predefined classes.")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, label = self.image_info[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation (train or standard)
        if self.transform:
            image = self.transform(image)

        return image, label



train_augment = transforms.Compose([
    transforms.Resize((100, 250)),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),  # Mild light/dark
    A.RandomGamma(gamma_limit=(80, 120), p=0.8),                                  # Avoid too dark/bright
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),                         # Contrast boost
    A.ToGray(p=0.2),                                                               # Sometimes grayscale
    A.Solarize(threshold=192, p=0.2),                                              # Less aggressive inversion
    A.InvertImg(p=0.2),                                                            # Rare inversion
    transforms.ToTensor,
])

# Standard Transformations for Validation & Test
standard_transform = transforms.Compose([
    transforms.Resize((100, 250)),
    transforms.ToTensor,
])

# Load dataset (no transform applied yet)
full_dataset = MovementDataset(
    json_file_path,
    augment=True,            # Default True; override per split
    augment_times=10,
     transform=None
)

# Split into Train, Validation, and Test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Set correct transforms after splitting
train_dataset.dataset.transform = train_augment
val_dataset.dataset.transform = standard_transform
test_dataset.dataset.transform = standard_transform

# Ensure only train uses augmentation
train_dataset.dataset.augment = True
val_dataset.dataset.augment = False
test_dataset.dataset.augment = False

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Total samples after repetition: {len(full_dataset)}")
print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

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
            nn.Linear(num_features, 512),
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
            torch.save(model.state_dict, "cnn_camera1_crop3_station5_new_augmentation_512.pth")

        scheduler.step

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = ClassificationModel(model_type="vgg", pretrained=True)

# Train
train_model(model, train_loader, val_loader, num_epochs=30, device=device)

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
model.load_state_dict(torch.load("cnn_camera1_crop3_station5_new_augmentation_512.pth"))
test_model(model, test_loader)
