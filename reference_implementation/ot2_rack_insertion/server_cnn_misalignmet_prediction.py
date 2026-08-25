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

#   **Model for 27-Class Classification**
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
            nn.Linear(1024, len(combined_labels))  #   27-class output
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

#   Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel(model_type='vgg')
model.load_state_dict(torch.load("path/to/cnn_model_misalignmet_predictor.pth", map_location=device))
model.to(device)
model.eval()

#   Transform
test_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])

#   Flask app
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

#   Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001)
