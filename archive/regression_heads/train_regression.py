import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights


# =====================================================
# Configuration
# =====================================================

# Paths
image_dirs = [
    os.path.join(BASE_DIR, 'external_both1/color_images/camera1'),
    os.path.join(BASE_DIR, 'external_both2/color_images/camera1'),

]
error_data_files = [
    os.path.join(BASE_DIR, 'external_both1/error_data.txt'),
     os.path.join(BASE_DIR, 'external_both2/error_data.txt'),

]

SAVE_MODEL_PATH = "depth_regressor_cbam_best.pth"
INPUT_IMG_SIZE = (250, 250)

BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 25
LEARNING_RATE = 0.0005


# =====================================================
# Dataset
# =====================================================

class StackedImageDataset(Dataset):
    def __init__(self, image_dirs, label_files, transform=None, augment_times=10):
        self.transform = transform
        self.image_info = []

        for img_dir, label_file in zip(image_dirs, label_files):
            labels = pd.read_csv(label_file, header=None, sep=" ", names=["dx"])

            for idx in range(len(labels)):
                img_path = os.path.join(img_dir, f"{idx + 1}.png")
                dx_value = float(labels.iloc[idx, 0])

                for _ in range(augment_times):
                    self.image_info.append((img_path, dx_value))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, dx_value = self.image_info[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([dx_value], dtype=torch.float32)


# =====================================================
# Transforms
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize(INPUT_IMG_SIZE),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4),
                           saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
    transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.5),
                                                   contrast=(0.5, 1.5))], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5),
                                                    sigma=(0.1, 2.0))], p=0.2),
    transforms.RandomApply([transforms.ColorJitter(brightness=(0.3, 1.7))], p=0.3),
    transforms.ToTensor,
])

val_test_transform = transforms.Compose([
    transforms.Resize(INPUT_IMG_SIZE),
    transforms.ToTensor,
])


# =====================================================
# Model: CBAM + VGG19 backbone
# =====================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super.__init__
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU,
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super.__init__
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super.__init__
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


class DepthRegressorCBAM(nn.Module):
    def __init__(self):
        super.__init__
        weights = VGG19_Weights.IMAGENET1K_V1
        self.backbone = vgg19(weights=weights)
        self.backbone.classifier = nn.Identity

        with torch.no_grad:
            dummy = torch.randn(1, 3, *INPUT_IMG_SIZE)
            feats = self.backbone.features(dummy)
            num_features = feats.view(1, -1).shape[1]

        self.cbam = CBAM(in_channels=512)
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU,
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =====================================================
# Training Setup
# =====================================================

full_dataset = StackedImageDataset(image_dirs, error_data_files, augment_times=10)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS//2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS//2)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = DepthRegressorCBAM.to(device)

criterion = nn.MSELoss
optimizer = optim.Adam(model.parameters, lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_loss = float("inf")


# =====================================================
# Training Loop
# =====================================================

for epoch in range(NUM_EPOCHS):
    model.train
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad
        loss = criterion(model(imgs), labels)
        loss.backward
        optimizer.step

        train_loss += loss.item * imgs.size(0)

    train_loss /= train_size
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}")

    model.eval
    val_loss = 0.0

    with torch.no_grad:
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_loss += criterion(model(imgs), labels).item * imgs.size(0)

    val_loss /= val_size
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict, SAVE_MODEL_PATH)

    scheduler.step


# =====================================================
# Testing
# =====================================================

model.load_state_dict(torch.load(SAVE_MODEL_PATH))
model.eval

test_loss = 0.0
with torch.no_grad:
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        test_loss += criterion(model(imgs), labels).item * imgs.size(0)

test_loss /= test_size
print(f"Test Loss: {test_loss:.6f}")
