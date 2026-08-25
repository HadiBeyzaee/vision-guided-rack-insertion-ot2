"""Archived variant. Originally named `save_cut_wrong_v2.py`.

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

# image_dir =  os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test1/color_images/camera1")
# output_dir = os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test1/color_images/camera1_cropped2")

image_dir =  os.path.join(BASE_DIR, "external_wrong4/color_images/camera1")
output_dir = os.path.join(BASE_DIR, "external_wrong4/color_images/camera1_cropped1")


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

#first
# top_margin = 400
# bottom_margin = 100

# left_margin = 450
# right_margin = 280

#wrong 1
top_margin = 410
bottom_margin = 100

left_margin = 480
right_margin = 280

# top_margin = 475
# bottom_margin = 100

# left_margin = 480
# right_margin = 280

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


# # middle
# top_margin = 440
# bottom_margin = 130

# left_margin = 470
# right_margin = 300

# # black forth crop
# top_margin = 445
# bottom_margin = 130

# left_margin = 480
# right_margin = 310

# Cropping margins
# #black third
# top_margin = 445
# bottom_margin = 130

# left_margin = 480
# right_margin = 340


#black second crop
# top_margin = 440
# bottom_margin = 130

# left_margin = 450
# right_margin = 310


# # black first crop
# top_margin = 440
# bottom_margin = 130

# left_margin = 480
# right_margin = 310


# # Cropping margins
# top_margin = 400
# bottom_margin = 150
# left_margin = 750
# right_margin = 300

# top_margin = 120
# bottom_margin = 150
# left_margin = 475
# right_margin = 300

# # Cropping margins
# top_margin = 430
# bottom_margin = 150
# left_margin = 475
# right_margin = 300


# top_margin = 435
# bottom_margin = 140

# left_margin = 750
# right_margin = 310


# top_margin = 425
# bottom_margin = 140

# left_margin = 450
# right_margin = 320

# top_margin = 370
# bottom_margin = 14

# left_margin = 380
# right_margin = 228


# top_margin = 360
# bottom_margin = 150

# left_margin = 445
# right_margin = 305

# top_margin = 370
# bottom_margin = 150

# left_margin = 465
# right_margin = 325


# # cut big
# top_margin = 420
# bottom_margin = 100

# left_margin = 400
# right_margin = 230

# # both 1
# top_margin = 440
# bottom_margin = 130

# left_margin = 460
# right_margin = 300

# # both 2
# top_margin = 450
# bottom_margin = 130

# left_margin = 470
# right_margin = 300

# # both 3
# top_margin = 445
# bottom_margin = 130

# left_margin = 460
# right_margin = 300
