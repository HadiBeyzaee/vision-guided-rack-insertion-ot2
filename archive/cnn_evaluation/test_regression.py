import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet101, resnet50, resnet18, vgg19
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg19
from torchvision.models import vgg16, vgg19, VGG19_Weights, ResNet18_Weights

# Custom dataset class
class StackedImageDataset(Dataset):
    def __init__(self, image_dirs, error_data_files, transform=None):
        self.image_info = []
        for image_dir, error_data_file in zip(image_dirs, error_data_files):
            error_data = pd.read_csv(error_data_file, header=None, sep=" ", names=["dx"])
            for i in range(len(error_data)):
                img_name = f"{i+1}.png"

                self.image_info.append((os.path.join(image_dir, img_name), error_data.iloc[i, :1].to_numpy(dtype='float32')))

        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, offsets = self.image_info[idx]
        image = Image.open(img_path) # Ensure the image is in PIL format

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(offsets, dtype=torch.float32), img_path


# Transformations
transform = transforms.Compose([
    transforms.Resize((250,100)),  # Resize the image to 224x224 while maintaining aspect ratio
    transforms.ToTensor,  # Convert PIL image to tensor
])



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
    def __init__(self, input_size=(250, 100)):
        super(DepthRegressorCBAM, self).__init__

        weights = VGG19_Weights.IMAGENET1K_V1
        self.feature_extractor = vgg19(weights=weights)
        self.feature_extractor.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.feature_extractor.classifier = nn.Identity

        # Calculate num_features dynamically
        with torch.no_grad:
            dummy_input = torch.randn(1, 3, *input_size)  # Batch size=1, channel=1, H=324, W=100
            features = self.feature_extractor.features(dummy_input)
            num_features = features.view(1, -1).size(1)  # Flattened size



        self.cbam = CBAM(in_channels=512)
        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU,
            nn.Linear(512, 1)  # Output (dx, dy, dtheta)
        )

    def forward(self, x):
        # Extract features using the selected backbone model
        if hasattr(self.feature_extractor, 'features'):
            # VGG19 branch
            x = self.feature_extractor.features(x)
        else:
            # ResNet18 branch
            x = self.feature_extractor(x)
            x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to [B, C, 1, 1] for CBAM compatibility

        x = self.cbam(x)  # Apply CBAM
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        return self.fc(x)



# Paths
image_dirs = [os.path.join(BASE_DIR, 'external_wrong1/color_images/camera1_cropped1')]
error_data_files = [os.path.join(BASE_DIR, 'external_wrong1/error_data.txt')]


# Dataset and DataLoader
test_dataset = StackedImageDataset(image_dirs, error_data_files, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = DepthRegressorCBAM.to(device)
model.load_state_dict(torch.load('train_reg_crop1_wrong3.pth'))

model.eval


# Function to calculate MSE for each component
def calculate_mse(loader, model):
    mse_loss = nn.MSELoss
    total_loss = 0.0
    total_loss_dx = 0.0

    all_actual_offsets = []
    all_predicted_offsets = []

    with torch.no_grad:
        for images, actual_offsets, _ in loader:
            images = images.to(device)
            actual_offsets = actual_offsets.to(device)
            outputs = model(images)

            # Convert dx, dy to mm and dtheta to degrees for MSE calculation
            actual_offsets[:, 0:1] *= 1000  # Convert meters to millimeters
            outputs[:, 0:1] *= 1000  # Convert meters to millimeters

            # Collect data for plotting
            all_actual_offsets.append(actual_offsets.cpu.numpy)
            all_predicted_offsets.append(outputs.cpu.numpy)

            # Calculate total MSE
            loss = mse_loss(outputs, actual_offsets)
            total_loss += loss.item * images.size(0)  # Multiply by batch size

            # Calculate MSE for each component
            total_loss_dx += mse_loss(outputs[:, 0], actual_offsets[:, 0]).item * images.size(0)

    mse = total_loss / len(loader.dataset)  # Divide by total number of samples
    mse_dx = total_loss_dx / len(loader.dataset)

    return mse, mse_dx, np.vstack(all_actual_offsets), np.vstack(all_predicted_offsets)

# Calculate and print the MSE for the test dataset
test_mse, test_mse_dx, actual_offsets, predicted_offsets = calculate_mse(test_loader, model)
print(f'Test Mean Squared Error (Total): {test_mse:.6f}')
print(f'Test Mean Squared Error (dx in mm): {test_mse_dx:.6f}')

# Plot comparisons for dx, dy, and dtheta
def plot_comparison(actual, predicted, label, unit, mse, save_as):
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, color='blue')
    plt.plot([actual.min, actual.max], [actual.min, actual.max], color='red', linestyle='--')
    plt.title(f'{label} Comparison\nTest Mean Squared Error ({unit}): {mse:.6f}')
    plt.xlabel(f'Actual {label} ({unit})')
    plt.ylabel(f'Predicted {label} ({unit})')
    plt.axis('equal')

    # Set axis limits
    if label in ['dx', 'dy']:
        plt.xlim([-60, 60])  # mm range
        plt.ylim([-60, 60])  # mm range
    elif label == 'dtheta':
        plt.xlim([-20, 20])  # degrees range
        plt.ylim([-20, 20])  # degrees range

    plt.grid
    plt.savefig(save_as)
    plt.show

# Plot results
plot_comparison(actual_offsets[:, 0], predicted_offsets[:, 0], 'dx', 'mm', test_mse_dx, 'dx_test_mm.png')
