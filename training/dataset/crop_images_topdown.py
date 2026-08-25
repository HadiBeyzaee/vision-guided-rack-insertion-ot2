"""Crop for the top-down camera used by the complete system.

Margins {top 240, bottom 320, left 500, right 200} - the same rectangle
complete_system/align_and_insert_cnn.py applies at run time to the `camera2`
stream.

crop_images.py holds the ViRA rectangles for the front camera instead. Use the
one matching the study whose dataset you are building; the two framings are not
interchangeable, and neither are models trained on them.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
from PIL import Image
import os

# (archived path variant removed)
# (archived path variant removed)

# (archived path variant removed)
# (archived path variant removed)

# (archived path variant removed)
# (archived path variant removed)

image_dir =  os.path.join(BASE_DIR, "paper_image1/color_images/camera2")
output_dir = os.path.join(BASE_DIR, "paper_image1/color_images/camera2_cropped1")


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# top_margin = 220
# bottom_margin = 250

# left_margin = 450
# right_margin = 150

top_margin = 240
bottom_margin = 320

left_margin = 500
right_margin = 200

# top_margin = 80
# bottom_margin = 450

# left_margin = 515
# right_margin = 250

# top_margin = 80
# bottom_margin = 450

# left_margin = 520
# right_margin = 255

# top_margin = 80
# bottom_margin = 450

# left_margin = 525
# right_margin = 240

# top_margin = 110
# bottom_margin = 450

# left_margin = 525
# right_margin = 240


# top_margin = 50
# bottom_margin = 450

# left_margin = 525
# right_margin = 240

# Function to crop the image based on specified margins
def crop_image(image, top=0, bottom=0, left=0, right=0):
    width, height = image.size
    # Define the coordinates for cropping based on the provided margins
    upper = top
    lower = height - bottom
    cropped_image = image.crop((left, upper, width - right, lower))
    return cropped_image

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(output_dir, image_file)

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Crop image with the specified margins
    cropped_image = crop_image(image, top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)

    # Save cropped image
    cropped_image.save(output_path)

print(f"Processed {len(image_files)} images and saved them to {output_dir}")

