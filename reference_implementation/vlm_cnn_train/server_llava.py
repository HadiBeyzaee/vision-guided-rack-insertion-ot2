import os
import subprocess
import tempfile
from flask import Flask, request, jsonify

# =========================================================
# User Configuration (Edit to match your environment)
# =========================================================

MODEL_PATH = "checkpoints/llava-v1.5-7b-opentron-lora"
MODEL_BASE = "pretrained/llava-v1.5-7b"

PORT = 5000


# =========================================================
# Label Set (27-class movement + rotation)
# =========================================================

movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

CLASS_LABELS = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]


# =========================================================
# Query prompt (must match training)
# =========================================================

QUERY = (
    "This cropped image shows the object position and orientation relative to the slot. "
    "What movement and what rotation are needed to align it properly?"
)


# =========================================================
# Flask app setup
# =========================================================

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    image_file = request.files["image"]

    # Save to a temporary file safely
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    # Run LLaVA inference
    command = [
        "python",
        "CONFIGURE_ME",
        "--model-path", MODEL_PATH,
        "--model-base", MODEL_BASE,
        "--image-file", image_path,
        "--query", QUERY
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip()

    # Cleanup image
    os.remove(image_path)

    # Extract final prediction from LLaVA output
    lines = output.split("\n")
    model_text = lines[-1] if lines else ""

    predicted_label = next(
        (label for label in CLASS_LABELS if label in model_text),
        "Unknown"
    )

    print("Predicted:", predicted_label)

    return jsonify({
        "predicted_label": predicted_label,
        "raw_output": model_text
    })


# =========================================================
# Start Server
# =========================================================

if __name__ == "__main__":
    print(f"LLaVA Server running on port {PORT}")
    print(f"Model: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=PORT)
