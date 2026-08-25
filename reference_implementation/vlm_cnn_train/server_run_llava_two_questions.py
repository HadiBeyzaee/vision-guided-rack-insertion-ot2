import os
import tempfile
import subprocess
from flask import Flask, request, jsonify

# ============================================================
# Configuration
# ============================================================

LLAVA_RUN_SCRIPT = "CONFIGURE_ME"

MODEL_PATH = "path/to/your/llava_model.pth"  # Update with your model path
MODEL_BASE = "path/to/your/llava_base_model"  # Update with your base model

PROMPT_MOVEMENT = (
    "This cropped image shows the object position and orientation relative to the slot. "
    "What movement and rotation are needed to align it properly?"
)

PROMPT_COARSE_FINE = (
    "For each of the following: vertical position, horizontal position, and rotation, "
    "is the misalignment large (coarse) or small (fine)?"
)

HOST = "0.0.0.0"
PORT = 5010


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


def run_llava(image_path: str, query: str) -> str:
    """Run LLaVA inference via subprocess and extract final answer."""
    command = [
        "python",
        LLAVA_RUN_SCRIPT,
        "--model-path", MODEL_PATH,
        "--model-base", MODEL_BASE,
        "--image-file", image_path,
        "--query", query,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if not result.stdout:
        return "Unknown"

    # Take last meaningful line (skip logs)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line and not line.startswith("["):
            return line

    return "Unknown"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    # Use a unique temp file to avoid collisions
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        movement = run_llava(image_path, PROMPT_MOVEMENT)
        granularity = run_llava(image_path, PROMPT_COARSE_FINE)

        return jsonify({
            "predicted_movement": movement,
            "predicted_granularity": granularity,
        })

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
