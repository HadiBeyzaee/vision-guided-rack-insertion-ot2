"""Crop variant that blurs everything outside a sharp horizontal band.

Keeps a sharp strip across the rack-slot boundary and blurs the rest before
cropping, to push the model towards the boundary geometry and away from
labware colour and pattern.

A different idea from crop_images.py, not a different set of numbers, which is
why it is kept. Analysis only; does not touch the robot.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
import os
from PIL import Image, ImageDraw, ImageFilter

# Paths
input_folder =  os.path.join(BASE_DIR, "opentron_middle_test/color_images/camera1")
output_folder = os.path.join(BASE_DIR, "opentron_middle_test/color_images/camera1_blured_cropped")
os.makedirs(output_folder, exist_ok=True)

# Fixed ROI (sharp zone)
x_start, x_end = 460, 970
y_top, y_bottom = 455, 510

# Crop margins
top_margin = 370
bottom_margin = 150
left_margin = 400
right_margin = 250

# Crop function
def crop_image_by_margin(img, top, bottom, left, right):
    width, height = img.size
    return img.crop((left, top, width - right, height - bottom))

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower.endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open and blur the image
        image = Image.open(input_path).convert("RGB")
        blurred = image.filter(ImageFilter.GaussianBlur(radius=15))

        # Create a binary mask for the sharp region
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x_start, y_top, x_end, y_bottom], fill=255)

        # Composite: keep ROI sharp, blur rest
        masked_image = Image.composite(image, blurred, mask)

        # Crop the result using flexible margins
        cropped = crop_image_by_margin(masked_image, top_margin, bottom_margin, left_margin, right_margin)

        # Save final image
        cropped.save(output_path)

print(f"All cropped + masked images saved to: {output_folder}")
