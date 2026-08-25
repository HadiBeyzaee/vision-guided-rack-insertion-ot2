import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19

# Coarse/Fine label mapping
cf_labels = ["coarse", "fine"]
combined_cf_labels = [
    f"{x} {y} {r}"
    for x in cf_labels
    for y in cf_labels
    for r in cf_labels
]
class_to_idx = {label: idx for idx, label in enumerate(combined_cf_labels)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Model definition
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.backbone = vgg19(pretrained=False)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, len(combined_cf_labels))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel()
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cnn_camera1_coarse_fine_crop2.pth"), map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor()
])

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_cf():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image = Image.open(request.files['image']).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': f"Image processing failed: {str(e)}"}), 500

    with torch.no_grad():
        logits = model(image)
        pred_idx = torch.argmax(logits, dim=1).item()
        movement_cf = idx_to_class[pred_idx]  # e.g., "coarse fine fine"

    return jsonify({
        "movement_cf": movement_cf
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4002)
