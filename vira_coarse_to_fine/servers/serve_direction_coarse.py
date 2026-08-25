"""ViRA coarse phase: which way to move. Port 5091.

Answers on the key "movement" with one of SIXTEEN labels - no "No Move", no
"No Rotate". That is deliberate: during the coarse phase the stop decision
belongs to serve_coarse_fine_flags.py, so this model is only ever asked for a
direction.

Repointing the client at a 27-class server here would terminate the loop on the
first No Move triple and skip the fine phase entirely.
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
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop2-nine-lora")
model_base = LLAVA_BASE

# Define label set (dx, dy, dtheta based)
movement_x = ["Move Up", "Move Down"]  # dx (vertical)
movement_y = ["Move Left", "Move Right"]  # dy (horizontal)
rotation = ["Rotate Clockwise", "Rotate clockwise", "Rotate Counterclockwise", "Rotate counterclockwise"]

combined_labels = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]

# Inference query
query_combined = (
    "This cropped image shows the object position and orientation relative to the slot. What movement and rotation are needed to align it properly?")

# Flask app setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Save uploaded image directly without resizing
    image = request.files['image']
    image_path = "/tmp/received_image.jpg"
    image.save(image_path)

    # Run LLaVA inference using the uploaded image
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query_combined
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    model_output = output_lines[-1] if output_lines else "Unknown"

    print("Raw Output:", result.stdout)
    print("Final Prediction:", model_output)

    # Match predicted label from known list
    predicted_label = next(
        (label for label in combined_labels if label in model_output), "Unknown"
    )

    return jsonify({"movement": predicted_label})

# Start server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5091)))
