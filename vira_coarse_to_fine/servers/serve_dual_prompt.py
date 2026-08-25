"""ViRA dual-prompt server: one adapter, both questions. Port 5010.

The deployment counterpart of training/vira_adapters/build_dual_prompt_dataset.py.
A single fine-tuned adapter is queried twice per frame - once for the correction
direction, once for the Coarse/Fine granularity - and both answers come back in
one reply, on the keys "predicted_movement" and "predicted_granularity".

This is the alternative to the split design that
align_coarse_to_fine_llava.py actually drives, where two separately fine-tuned
adapters answer on ports 5091 and 5012. One adapter instead of two, at the cost
of two sequential generations per frame.

NOTE: the reply keys here are NOT the "movement" / "movement_cf" that the
control loop reads. Driving the loop from this server needs either a client
change or a key rename. See docs/inference_servers.md.
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

# === Model paths ===
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop2-multi-8class-lora")
model_base = LLAVA_BASE

# === Prompts ===
prompt_movement = (
    "This cropped image shows the object position and orientation relative to the slot. "
                        "What movement and rotation are needed to align it properly?"
)

prompt_coarse_fine = (
    "For each of the following: vertical position, horizontal position, and rotation, "
        "is the misalignment large (coarse) or small (fine)?"
)

# === Flask app ===
app = Flask(__name__)

def run_llava(image_path, query):
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.startswith("["):
            return line.strip()
    return "Unknown"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = "/tmp/received_image.jpg"
    image.save(image_path)

    # === Run both queries ===
    pred_movement = run_llava(image_path, prompt_movement)
    pred_granularity = run_llava(image_path, prompt_coarse_fine)

    print("\nMovement:", pred_movement)
    print("Coarse/Fine:", pred_granularity)

    return jsonify({
        "predicted_movement": pred_movement,
        "predicted_granularity": pred_granularity
    })

# === Run the server ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5010)))
