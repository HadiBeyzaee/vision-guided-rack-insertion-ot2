import os
from flask import Flask, request, jsonify
import subprocess

# Model configuration
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-offset-amount-lora")
model_base = LLAVA_BASE

# Yes/No Offset Query (dx, dy, dtheta)
query_offset = (
    "Is the offset between the silver rack holder and the black rack large in any of these aspects: "
    "1) Vertical alignment, 2) Horizontal alignment, 3) Rotation? "
    "Answer with: yes/no yes/no yes/no"
)

# Flask app setup
app = Flask(__name__)

@app.route('/predict_amount', methods=['POST'])
def predict_offset_flags():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = "/tmp/received_image.jpg"
    image.save(image_path)

    # Run LLaVA inference
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query_offset
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    output_lines = result.stdout.strip().split("\n")
    raw_output = output_lines[-1] if output_lines else "Unknown"
    print("Raw Output:\n", result.stdout)

    # Parse last valid line with 3-token yes/no response
    prediction = "Unknown"
    for line in reversed(output_lines):
        parts = line.strip().lower().split()
        if len(parts) == 3 and all(p in ["yes", "no"] for p in parts):
            prediction = " ".join(parts)
            break

    return jsonify({
        "amount": prediction  # Format: "yes no yes"
    })

# Run server on port 5001
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005)
