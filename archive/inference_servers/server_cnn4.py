import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, vgg19

movement_y = ["Move Left", "Move Right", "No Move"]
movement_x = ["Move Up", "Move Down", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

# Ensure same order
combined_labels = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]



# Mapping label index
class_to_idx = {label: idx for idx, label in enumerate(combined_labels)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# CBAM components for attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Modified Model for 27-Class Classification
class ClassificationModel(nn.Module):
    def __init__(self, model_type='resnet', pretrained=True):
        super(ClassificationModel, self).__init__()

        if model_type == 'resnet':
            self.backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_type == 'vgg':
            self.backbone = vgg19(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("model_type must be 'resnet' or 'vgg'")

        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, len(combined_labels))  # 27-class output
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel(model_type='resnet')
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cnn_camera2_crop2_dxyt_1234.pth"), map_location=device))
model.to(device)
model.eval()

# Transform
test_transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
])

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = "/tmp/temp_input_image.jpg"
    image.save(image_path)

    # Preprocess
    try:
        img = Image.open(image_path).convert("RGB")
        img = test_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {e}'}), 500

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, pred_idx = torch.max(outputs, 1)
        predicted_label = idx_to_class[pred_idx.item()]

    return jsonify({'movement': predicted_label})

# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4400)
