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

image_dir =  os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera2")
output_dir = os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera2_cropped2")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
# Cropping margins

# top_margin = 170
# bottom_margin = 410

# left_margin = 460
# right_margin = 310

# high crop
top_margin = 100
bottom_margin = 410

left_margin = 380
right_margin = 230

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
