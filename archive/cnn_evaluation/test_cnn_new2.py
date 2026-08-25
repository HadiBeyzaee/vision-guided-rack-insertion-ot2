import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, vgg19
import matplotlib.pyplot as plt
from tqdm import tqdm

# # Define test image folder & labels file
# test_folder =     os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera1_cropped")
# error_data_file = os.path.join(BASE_DIR, "opentron_station1_test/error_data.txt")

test_folder = os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test1/color_images/camera1_cropped2")
error_data_file = os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test1/error_data.txt")

# Generate all 27 class labels in the correct order
movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

# # Generate all possible labels (27 total)
# combined_labels = sorted([
# f"{x}, {y}, {r}"
# for x in movement_x
# for y in movement_y
# for r in rotation
# ])

combined_labels = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]


# Create label index mapping
class_to_idx = {label: idx for idx, label in enumerate(combined_labels)}


def classify_movement(dx, dy):
    move_x, move_y = "No Move", "No Move"

    if dx > 0.0005:
        move_x = "Move Down"
    elif dx < -0.0005:
        move_x = "Move Up"

    if dy > 0.0005:
        move_y = "Move Right"
    elif dy < -0.0005:
        move_y = "Move Left"

    return move_x, move_y

def classify_rotation(dtheta):
    if -0.2 <= dtheta <= 0.2:
        return "No Rotate"
    elif dtheta > 0.2:
        return "Rotate Clockwise"
    elif dtheta < -0.2:
        return "Rotate Counterclockwise"


# Function to generate full label
def classify_combined(dx, dy, dtheta):
    move_x, move_y = classify_movement(dx, dy)
    rotation_label = classify_rotation(dtheta)

    return f"{move_x}, {move_y}, {rotation_label}"

# Load ground truth labels from `error_data.txt`
ground_truth = {}
with open(error_data_file, "r") as f:
    for idx, line in enumerate(f.readlines):
        values = list(map(float, line.strip.split))
        if len(values) < 3:
            continue  # Skip invalid rows
        dx, dy, dtheta = values
        label = classify_combined(dx, dy, dtheta)
        if label in class_to_idx:
            ground_truth[f"{idx+1}.png"] = label  # Match image filename to label
        else:
            print(f"Warning: Label '{label}' not found in predefined classes.")

# Define test transformations (should match training pipeline)
test_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor,
])

# Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class ClassificationModel(nn.Module):
    def __init__(self, model_type='resnet'):
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
            raise ValueError("model_type must be 'resnet' or 'vgg'")

        # self.fc = nn.Sequential(
        # nn.Linear(num_features, 1024),
        # nn.ReLU,
        # nn.Linear(1024, len(combined_labels))  # 27-class output
        # )

        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU,
            nn.Linear(1024, 512), # 27-class output
            nn.ReLU,
            nn.Linear(512, len(combined_labels))  # 27-class output
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

# Load the trained model
model = ClassificationModel(model_type="vgg")  # Change to "resnet" if needed
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cnn_camera1_crop2_aug_wrong2.pth"), map_location=device))
model.to(device)
model.eval  # Set to evaluation mode

# Get list of test images
image_files = sorted(
    [f for f in os.listdir(test_folder) if f.endswith((".png", ".jpg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)


print(f"\nFound {len(image_files)} test images. Running inference...\n")

# Accuracy counters
correct_predictions = 0
total_images = 0

# Per-class accuracy tracking
correct_counts = {label: 0 for label in combined_labels}
total_counts = {label: 0 for label in combined_labels}

# Process each image
for img_file in tqdm(image_files, desc="Processing Test Images"):
    image_path = os.path.join(test_folder, img_file)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)  # Apply transformations & add batch dimension

    # Run inference
    with torch.no_grad:
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)  # Get predicted class index

    predicted_label = combined_labels[predicted_class.item]

    # Get ground truth label
    true_label = ground_truth.get(img_file, "Unknown")

    # Compare prediction with ground truth
    is_correct = predicted_label == true_label
    if is_correct:
        correct_predictions += 1
    total_images += 1

    # Update per-class accuracy
    if true_label in total_counts:
        total_counts[true_label] += 1
        if is_correct:
            correct_counts[true_label] += 1

    # Print Result
    print(f"Image: {img_file}")
    print(f"Ground Truth: {true_label}")
    print(f"Predicted Class: {predicted_label} {'' if is_correct else ''}")
    print("---------------------------------------")

# Compute & Print Overall Accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"\n**Final Test Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})**")

# Compute per-class accuracy
accuracy_per_label = {
    label: (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 for label in combined_labels
}

# Plot Accuracy for Each Class
plt.figure(figsize=(12, 6))
plt.bar(combined_labels, [accuracy_per_label[label] for label in combined_labels], color="blue")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy for Each Movement & Rotation Class")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)
plt.savefig("best_27_class_model.png")
plt.show
