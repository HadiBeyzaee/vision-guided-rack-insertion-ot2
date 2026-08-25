"""Per-axis translation server. Port 5001, route /predict_translation.

Answers on the key "movement" with the translation decision only. Pairs with
serve_rotation.py to drive legacy/insert_with_llava_per_axis.py, which asks
about translation and rotation separately.

Belongs to the pre-OT-2 rig, not to either study here.
"""

import os
from flask import Flask, request, jsonify
import subprocess

# --- Paths and port (override in your shell or a .env file) ------------
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------

model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-twenty-seven-full-augmented-dxyt-lora")
model_base = LLAVA_BASE

movement_y = ["Move Down", "Move Up", "No Move"]  # dx (Up/Down)
movement_x = ["Move Left", "Move Right", "No Move"]  # dy (Left/Right)

translation_labels = [f"{x}, {y}" for x in movement_x for y in movement_y]

# Translation query
query_translation = (
"How should the black rack (bottom) move to align with the silver holder above? Choose one translation (Up, Down, Left, Right), or No Move if already aligned."
)

app = Flask(__name__)

@app.route('/predict_translation', methods=['POST'])
def predict_translation():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = "/tmp/received_image.jpg"
    image.save(image_path)

    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query_translation
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    model_output = output_lines[-1] if output_lines else "Unknown"
    print("Translation Raw Output:", result.stdout)

    predicted_label = next(
    (label for label in translation_labels if label.strip() in model_output.strip()),
    "Unknown"
)
    return jsonify({"movement": predicted_label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))  # Port 5001 for dx/dy
