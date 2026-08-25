"""UUID rename that writes the {original_name, new_name, row_number} schema.

There are TWO mapping schemas in this project, and the builders are not
interchangeable between them:

    rename_images.py            writes  original_filename / new_filename / row_index
                                read by dataset/build_llava_dataset.py
    rename_images_rowmapping.py writes  original_name / new_name / row_number
                                read by every builder in vira_adapters/

Running the wrong one gives a KeyError deep inside the builder rather than
anything that names the real problem. Match the renamer to the builder you
intend to feed.
"""

import os
import json
import uuid
import re
import shutil  # For copying files safely

# Define paths
txt_file_path =         os.path.join(BASE_DIR, "camera2_dxyt4/error_data_align.txt")
original_image_folder = os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align_new")
new_image_folder =      os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align_new_renamed")
output_mapping_json =   os.path.join(BASE_DIR, "camera2_dxyt4/color_images/image_row_mapping_camera2_align_new_renamed.json")

# (archived path variant removed)
# (archived path variant removed)
# (archived path variant removed)
# (archived path variant removed)

# Ensure new folder existss
os.makedirs(new_image_folder, exist_ok=True)

# Function for correct numerical sorting (fixes "1.png" vs "10.png" issue)
def numerical_sort(filename):
    return [int(text) if text.isdigit else text for text in re.split(r'(\d+)', filename)]

# Read the movement data from the TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Get sorted image files (numerical order)
image_files = sorted(
    [f for f in os.listdir(original_image_folder) if f.endswith('.png') or f.endswith('.jpg')],
    key=numerical_sort
)

# Ensure the number of images matches the number of labels
if len(image_files) != len(movement_lines):
    print(f"Warning: {len(image_files)} images vs {len(movement_lines)} movement records!")

# Create a mapping dictionary
image_mapping = {}

# Process each image, assign a unique name, and track row reference
for index, img_file in enumerate(image_files):
    original_path = os.path.join(original_image_folder, img_file)

    # Generate a unique ID
    unique_id = uuid.uuid4.hex  # Example: "86090d8bf5114265b63cd9649c2fbc35"
    new_filename = f"{unique_id}.png"
    new_path = os.path.join(new_image_folder, new_filename)

    # Copy image with the new name (keeping original intact)
    shutil.copy2(original_path, new_path)

    # Store mapping (original name -> unique name + row number)
    image_mapping[unique_id] = {
        "original_name": img_file,
        "new_name": new_filename,
        "row_number": index + 1  # Matches row in TXT file (1-based index)
    }

# Save mapping to JSON file
with open(output_mapping_json, "w") as json_file:
    json.dump(image_mapping, json_file, indent=4)

print(f"\nImage renaming complete. Renamed images are saved in: {new_image_folder}")
print(f"Mapping file saved: {output_mapping_json}")
