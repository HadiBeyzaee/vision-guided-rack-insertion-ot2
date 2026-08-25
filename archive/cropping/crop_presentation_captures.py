"""Archived variant. Originally named `save_cut_wrong.py`.

"wrong" in the original name is shorthand for **wrong grasp** - the
deliberately off-centre grasp used to generate residual x error. It never
meant that the code was defective. Renamed here so a reader does not have to
guess.
"""

import os

# --- Archived variant. Paths parameterised; otherwise unmodified. ------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
from PIL import Image
import os

image_dir =  os.path.join(BASE_DIR, "presentation_images/color_images/camera2")
output_dir = os.path.join(BASE_DIR, "presentation_images/color_images/camera2_cropped1")

# image_dir =  os.path.join(BASE_DIR, "camera2_dxyt1/color_images/camera2_align_new_renamed")
# output_dir = os.path.join(BASE_DIR, "camera2_dxyt1/color_images/camera2_align_renamed_cropped1")

# image_dir =  os.path.join(BASE_DIR, "slot6_grasp_base_offset1/color_images/camera1_align_new_renamed")
# output_dir = os.path.join(BASE_DIR, "slot6_grasp_base_offset1/color_images/camera1_align_renamed_cropped4")

# image_dir =  os.path.join(BASE_DIR, "camera2_dxyt1/color_images/camera2_renamed")
# output_dir = os.path.join(BASE_DIR, "camera2_dxyt1/color_images/camera2_renamed_cropped1")


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# top_margin = 220
# bottom_margin = 250

# left_margin = 450
# right_margin = 150

top_margin = 300
bottom_margin = 300

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

