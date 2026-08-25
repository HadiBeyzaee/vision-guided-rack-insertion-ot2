import os
import subprocess
import re
from tqdm import tqdm
from PIL import Image

# === PATHS ===
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop1-coarse-fine-new2-lora")
model_base = LLAVA_BASE
fail_folder = os.path.join(BASE_DIR, "test_opentron_cf6/color_images/camera1_cropped")
error_data_file = os.path.join(BASE_DIR, "test_opentron_cf6/error_data.txt")

# === LOAD GROUND TRUTH FROM TXT ===
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

def classify_coarse_fine(dx, dy, dtheta):
    label_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    label_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    label_theta = "Coarse" if abs(dtheta) > 0.9 else "Fine"
    return f"{label_dx}, {label_dy}, {label_theta}"

def get_ground_truth(image_filename):
    try:
        index = int(os.path.splitext(image_filename)[0]) - 1
        dx, dy, dtheta = map(float, movement_lines[index].split)
        return classify_coarse_fine(dx, dy, dtheta)
    except Exception as e:
        print(f"Skipping {image_filename}: {e}")
        return None

def run_llava(image_path, query):
    resized_path = "/tmp/resized_input_image.jpg"
    image = Image.open(image_path).convert("RGB")  # Just load + convert to RGB, no resize
    image.save(resized_path)

    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", resized_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip


# === QUERY USED IN TRAINING/INFERENCE ===
query = (
    "In each direction, is the misalignment coarse or fine?"
)

# === ACCURACY METRICS ===
correct = 0
total = 0
label_stats = {}

# === MAIN LOOP ===
image_files = sorted(
    [f for f in os.listdir(fail_folder) if f.endswith(".png") or f.endswith(".jpg")],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)

for img_file in tqdm(image_files, desc="Evaluating"):
    img_path = os.path.join(fail_folder, img_file)
    gt_label = get_ground_truth(img_file)
    if not gt_label:
        continue

    pred_output = run_llava(img_path, query)
    predicted_label = next((l for l in ["Fine, Fine, Fine", "Coarse, Fine, Fine", "Fine, Coarse, Fine",
                                        "Fine, Fine, Coarse", "Coarse, Coarse, Coarse", "Coarse, Fine, Coarse",
                                        "Fine, Coarse, Coarse", "Coarse, Coarse, Fine", "Fine, Fine, Fine"]
                            if l in pred_output), pred_output)

    is_correct = predicted_label.strip == gt_label.strip
    total += 1
    correct += int(is_correct)

    # Update stats
    label_stats.setdefault(gt_label, {"total": 0, "correct": 0})
    label_stats[gt_label]["total"] += 1
    if is_correct:
        label_stats[gt_label]["correct"] += 1

    print(f"\n{img_file}")
    print(f"GT:   {gt_label}")
    print(f"PRED: {predicted_label}")
    print("CORRECT" if is_correct else "INCORRECT")
    print("-" * 40)

# === SUMMARY ===
final_acc = (correct / total) * 100 if total > 0 else 0
print(f"\nFINAL ACCURACY: {final_acc:.2f}% ({correct}/{total})")

print("\nPER LABEL ACCURACY:")
for label, stat in label_stats.items:
    acc = 100 * stat["correct"] / stat["total"]
    print(f"{label:<30} {acc:.2f}% ({stat['correct']}/{stat['total']})")
