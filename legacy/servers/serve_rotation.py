"""Per-axis rotation server. Port 5002, route /predict_rotation.

Answers on the key "rotation" with the yaw decision only. The other half of the
pair that drives legacy/insert_with_llava_per_axis.py.

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

model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-twenty-seven-crop1-augmented-lora")
model_base = LLAVA_BASE

rotation_labels = ["Rotate Clockwise", "Rotate clockwise", "Rotate Counterclockwise" , "Rotate counterclockwise", "No Rotate"]

query_rotation = (
    """How should the black rack (bottom) rotate to align with the silver holder above? Choose from: Rotate Clockwise, Rotate Counterclockwise, or No Rotate."""
)

app = Flask(__name__)

@app.route('/predict_rotation', methods=['POST'])
def predict_rotation():
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
        "--query", query_rotation
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    model_output = output_lines[-1] if output_lines else "Unknown"
    print("Rotation Raw Output:", result.stdout)

    predicted_label = next((label for label in rotation_labels if label in model_output), "Unknown")
    return jsonify({"rotation": predicted_label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))  # Port 5002 for dtheta
