"""ViRA phase switch: is each axis still coarse, or now fine. Port 5012.

Answers on the key "movement_cf" with a Coarse/Fine triple. When all three come
back Fine, the control loop latches to the fine direction server and drops its
step sizes from 7 mm / 0.8 deg to 0.9 mm / 0.1 deg.
"""

import os
from flask import Flask, request, jsonify
import subprocess

# --- Paths and port (override in your shell or a .env file) ------------
# LLAVA_REPO   : checkout of haotian-liu/LLaVA providing llava/eval/run_llava.py
# LLAVA_BASE   : base llava-v1.5-7b weights
# CHECKPOINT_DIR: directory holding the fine-tuned LoRA adapters
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------

# Model configuration
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop1-coarse-fine-new2-lora")
model_base = LLAVA_BASE

# Coarse/Fine label combinations
coarse_fine_labels = [
    f"{dx}, {dy}, {theta}"
    for dx in ["Coarse", "Fine"]
    for dy in ["Coarse", "Fine"]
    for theta in ["Coarse", "Fine"]
]

# Prompt used during training/inference
query_cf = (
   "In each direction, is the misalignment coarse or fine?"
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = "/tmp/received_image.jpg"
    image.save(image_path)

    # Run LLaVA
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query_cf
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    model_output = output_lines[-1] if output_lines else "Unknown"

    print("\nRaw Output:\n", result.stdout)
    print("Final Prediction:", model_output)

    # Find matching coarse/fine label
    predicted_label = next(
        (label for label in coarse_fine_labels if label in model_output),
        "Unknown"
    )

    return jsonify({"movement_cf": predicted_label})

# Start server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5012)))
