from PIL import Image
import os


IMAGE_DIR = "/your/local/path/here"
OUTPUT_DIR = "/your/local/output/path"

# Margins for cropping (coarse)
CROP_MARGINS = {
    "top": 440,
    "bottom": 80,
    "left": 380,
    "right": 230,
}

# # Margins for cropping (fine)
# CROP_MARGINS = {
#     "top": 450,
#     "bottom": 140,
#     "left": 460,
#     "right": 310,
# }

os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_image(image, top=0, bottom=0, left=0, right=0):
    """Crop margins from an image."""
    width, height = image.size
    return image.crop((left, top, width - right, height - bottom))

def main():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")]

    for img_name in image_files:
        input_path = os.path.join(IMAGE_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, img_name)

        with Image.open(input_path) as img:
            img = img.convert("RGB")
            cropped = crop_image(img, **CROP_MARGINS)
            cropped.save(output_path)

    print(f"Processed {len(image_files)} images → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
