import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, vgg19
import matplotlib.pyplot as plt
from tqdm import tqdm

# Coarse/Fine thresholds
def classify_cf(value, axis):
    if axis == 'dtheta':
        return 'coarse' if abs(value) > 1.0 else 'fine'
    else:
        return 'coarse' if abs(value) > 0.005 else 'fine'

# # # Define test image folder & labels file
# test_folder =     os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera1_cropped3")
# error_data_file = os.path.join(BASE_DIR, "opentron_station1_test/error_data.txt")

test_folder = os.path.join(BASE_DIR, "test_opentron_cf6/color_images/camera1_cropped3")
error_data_file = os.path.join(BASE_DIR, "test_opentron_cf6/error_data.txt")

model_path = os.path.join(BASE_DIR, "cnn_camera1_coarse_fine_crop3.pth")

# All coarse/fine combinations (8 total)
cf_labels = ["coarse", "fine"]
combined_labels = [
    f"{dx} {dy} {dtheta}"
    for dx in cf_labels
    for dy in cf_labels
    for dtheta in cf_labels
]

# Create label maps
class_to_idx = {label: idx for idx, label in enumerate(combined_labels)}
idx_to_class = {v: k for k, v in class_to_idx.items}

# Ground truth mapping
ground_truth = {}
with open(error_data_file, "r") as f:
    for idx, line in enumerate(f.readlines):
        values = list(map(float, line.strip.split))
        if len(values) != 3:
            continue
        dx, dy, dtheta = values
        dx_flag = classify_cf(dx, 'dx')
        dy_flag = classify_cf(dy, 'dy')
        dtheta_flag = classify_cf(dtheta, 'dtheta')
        label = f"{dx_flag} {dy_flag} {dtheta_flag}"
        ground_truth[f"{idx+1}.png"] = label

# Transform pipeline
test_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor,
])

# Define model
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class ClassificationModel(nn.Module):
    def __init__(self, model_type='vgg'):
        super(ClassificationModel, self).__init__
        if model_type == 'resnet':
            self.backbone = resnet18(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity
        elif model_type == 'vgg':
            self.backbone = vgg19(pretrained=False)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity
        else:
            raise ValueError("Invalid model type")
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU,
            nn.Linear(1024, len(combined_labels))  # 8 output classes
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Load model
model = ClassificationModel(model_type="vgg")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval

# Evaluation
image_files = sorted(
    [f for f in os.listdir(test_folder) if f.endswith((".png", ".jpg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)

correct = 0
total = 0
correct_counts = {label: 0 for label in combined_labels}
total_counts = {label: 0 for label in combined_labels}

print(f"\nFound {len(image_files)} test images. Running inference...\n")

for img_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(test_folder, img_file)
    true_label = ground_truth.get(img_file)
    if true_label is None:
        print(f"No ground truth for {img_file}")
        continue

    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad:
        logits = model(image)
        pred_idx = torch.argmax(logits, dim=1).item
        pred_label = idx_to_class[pred_idx]

    is_correct = pred_label == true_label
    if is_correct:
        correct += 1
        correct_counts[true_label] += 1
    total += 1
    total_counts[true_label] += 1

    print(f"{img_file} | GT: {true_label} | Pred: {pred_label} {'' if is_correct else ''}")

# Final Accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Per-Class Accuracy
accuracy_per_label = {
    label: (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
    for label in combined_labels
}

# Plot
plt.figure(figsize=(10, 5))
plt.bar(combined_labels, [accuracy_per_label[label] for label in combined_labels])
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Coarse/Fine Class (dx dy dtheta)")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.tight_layout
plt.savefig("coarse_fine_accuracy.png")
plt.show
