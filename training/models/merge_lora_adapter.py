"""Merge a LoRA adapter into the base weights and save the result.

Not required by anything in this repository: every server loads the adapter
alongside the base model with --model-base. Useful when you want a single
self-contained checkpoint to hand to another tool.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and LoRA adapter
base_model_path = LLAVA_BASE
lora_model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop1-four-five-six-121234-lora")
merged_save_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-merged-lora")

# Load and merge
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload

# Save merged model
model.save_pretrained(merged_save_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_save_path)

print(f"Merged model saved to {merged_save_path}")
