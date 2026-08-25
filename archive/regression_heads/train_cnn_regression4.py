import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, vgg19, resnet18
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19, VGG19_Weights, ResNet18_Weights


class StackedImageDataset(Dataset):
    def __init__(self, image_dirs, error_data_files, transform=None, augment_times=10):
        self.image_info = []
        self.transform = transform
        self.augment_times = augment_times

        for image_dir, error_data_file in zip(image_dirs, error_data_files):
            # Read dx and dy from the error file
            error_data = pd.read_csv(error_data_file, header=None, sep=" ", names=["dx", "dy"])
            for i in range(len(error_data)):
                img_name = f"{i+1}.png"
                dx = float(error_data.iloc[i]["dx"])
                dy = float(error_data.iloc[i]["dy"])
                for _ in range(augment_times):
                    self.image_info.append((os.path.join(image_dir, img_name), [dx, dy]))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, offsets = self.image_info[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(offsets, dtype=torch.float32)  # shape: [2]


# Define realistic lighting augmentation transformations with adjustments to prevent black images
train_transform = transforms.Compose([
    transforms.Resize((100,250)),

    # Simulate various lighting conditions by adjusting brightness and contrast
    transforms.ColorJitter(
        brightness=(0.6, 1.4),  # Randomly increase/decrease brightness
        contrast=(0.6, 1.4),    # Randomly increase/decrease contrast
        saturation=(0.8, 1.2),  # Slightly vary saturation to simulate different light sources
        hue=(-0.1, 0.1)),       # Slightly vary hue to simulate different color temperatures

    # Randomly apply shadows or highlights by altering parts of the image
    transforms.RandomApply([
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5))
    ], p=0.5),

    # Add subtle blurring to simulate light scattering or focus issues
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
    ], p=0.2),

    # Randomly simulate underexposure or overexposure conditions
    transforms.RandomApply([
        transforms.ColorJitter(brightness=(0.3, 1.7))
    ], p=0.3),

    # Convert the image to a tensor
    transforms.ToTensor,
])

# For validation and testing, only apply necessary transformations without augmentation
val_test_transform = transforms.Compose([
    transforms.Resize((100,250)),  # Resize the image
    transforms.ToTensor,  # Convert PIL image to tensor
])

# Paths
image_dirs = [
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base1/color_images/camera1_cropped2'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base2/color_images/camera1_cropped2'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base3/color_images/camera1_cropped2'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base4/color_images/camera1_cropped2'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base5/color_images/camera1_cropped2'),

    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base1/color_images/camera1_align_cropped2_name'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base2/color_images/camera1_align_cropped2_name'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base3/color_images/camera1_align_cropped2_name'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base4/color_images/camera1_align_cropped2_name'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base5/color_images/camera1_align_cropped2_name'),
]
error_data_files = [
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base1/error_data_new.txt'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base2/error_data_new.txt'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base3/error_data_new.txt'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base4/error_data_new.txt'),
    os.path.join(BASE_DIR, 'opentron_wrong_grasp_base5/error_data_new.txt'),

    os.path.join(BASE_DIR, "opentron_wrong_grasp_base1/error_data_align.txt"),
    os.path.join(BASE_DIR, "opentron_wrong_grasp_base2/error_data_align.txt"),
    os.path.join(BASE_DIR, "opentron_wrong_grasp_base3/error_data_align.txt"),
    os.path.join(BASE_DIR, "opentron_wrong_grasp_base4/error_data_align.txt"),
    os.path.join(BASE_DIR, "opentron_wrong_grasp_base5/error_data_align.txt"),
]

# Dataset and DataLoader
full_dataset = StackedImageDataset(image_dirs, error_data_files, transform=None, augment_times=20)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Assign transformations based on dataset type
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


# CBAM components for attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU,
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class DepthRegressorCBAM(nn.Module):
    def __init__(self, input_size=(100, 250)):
        super(DepthRegressorCBAM, self).__init__

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.feature_extractor = resnet18(weights=weights)

        # Remove the final classification layer
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity

        # CBAM applied after feature extraction (output: [B, 512, H, W])
        self.cbam = CBAM(in_channels=512)

        # Determine output feature size after CBAM
        with torch.no_grad:
            dummy_input = torch.randn(1, 3, *input_size)
            x = self.feature_extractor(dummy_input)
            x = x.unsqueeze(-1).unsqueeze(-1) if x.ndim == 2 else x  # Handle [B, 512] vs [B, 512, H, W]
            x = self.cbam(x)
            num_cbam_features = x.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(num_cbam_features, 512),
            nn.ReLU,
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 512]
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        x = self.cbam(x)
        x = torch.flatten(x, 1)
        return self.fc(x)




# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f"Using device: {device}")

model = DepthRegressorCBAM.to(device)

# Loss and optimizer
criterion = nn.MSELoss
optimizer = optim.Adam(model.parameters, lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Lists to store losses
train_losses = []
val_losses = []

# Training the model
num_epochs = 30
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train
    running_loss = 0.0
    for images, offsets in train_loader:
        images = images.to(device, dtype=torch.float32)  # Ensure the input is float32
        offsets = offsets.to(device)

        optimizer.zero_grad
        outputs = model(images)
        loss = criterion(outputs, offsets)
        loss.backward
        optimizer.step

        running_loss += loss.item * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.11f}')

    # Validation
    model.eval
    val_loss = 0.0
    with torch.no_grad:
        for images, offsets in val_loader:
            images = images.to(device, dtype=torch.float32)  # Ensure the input is float32
            offsets = offsets.to(device)
            outputs = model(images)
            loss = criterion(outputs, offsets)
            val_loss += loss.item * images.size(0)
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.11f}')

    # Save the model if validation loss decreases
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict, 'train_regression_data1234_merged2_resnet.pth')

    scheduler.step


# Load the best model for testing
model.load_state_dict(torch.load('train_regression_data1234_merged2_resnet.pth'))
model.eval

# Testing
test_loss = 0.0
with torch.no_grad:
    for images, offsets in test_loader:
        images = images.to(device, dtype=torch.float32)  # Ensure the input is float32
        offsets = offsets.to(device)
        outputs = model(images)
        loss = criterion(outputs, offsets)
        test_loss += loss.item * images.size(0)
test_loss /= len(test_dataset)
print(f'Test Loss: {test_loss:.11f}')
