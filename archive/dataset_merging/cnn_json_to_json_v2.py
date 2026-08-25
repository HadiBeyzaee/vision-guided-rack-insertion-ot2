import os
import json

# Define input and output JSON paths
input_json = os.path.join(BASE_DIR, "augmented_crop1_wrong_wrong_merged.json")
output_json = os.path.join(BASE_DIR, "cnn_augmented_crop1_wrong_wrong_merged.json")

# Load the existing JSON file
with open(input_json, "r") as f:
    data = json.load(f)

# Extract image paths and labels
image_label_data = []
for entry in data:
    image_path = entry["image"]
    label_text = entry["conversations"][1]["value"]  # Get GPT label

    # Append to the new dataset
    image_label_data.append({
        "image": image_path,
        "label": label_text
    })

# Save the new JSON file
with open(output_json, "w") as f:
    json.dump(image_label_data, f, indent=4)

print(f"\nNew JSON file saved: {output_json} ")
