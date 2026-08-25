import os

# --- Archived variant. Paths parameterised; otherwise unmodified. ------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import json
from PIL import Image
import os

llava_json_path = os.path.join(BASE_DIR, "camera1_crop2_coarse_fine_updated_updated_merged.json")
qwen_json_output_path = os.path.join(BASE_DIR, "camera1_crop2_coarse_fine_updated_updated_merged_qwen_format.json")

# Load LLaVA-style data
with open(llava_json_path, "r") as f:
    llava_data = json.load(f)

qwen_data = []

system_message = (
    "You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, "
    "and interpreting visual data. Your task is to help align an object precisely into a slot using image guidance."
)

for sample in llava_data:
    img_path = sample["image"]

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    conv = sample["conversations"]
    if len(conv) != 2:
        print(f"Skipping {sample['id']}: Invalid conversation length")
        continue

    question = conv[0]["value"].replace("<image>\n", "").strip
    answer = conv[1]["value"].strip

    qwen_format = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
    }
    qwen_data.append(qwen_format)

# Save to JSON
with open(qwen_json_output_path, "w") as f:
    json.dump(qwen_data, f, indent=4)

print(f"Saved {len(qwen_data)} samples in Qwen2-VL format to: {qwen_json_output_path}")
