"""LoRA fine-tuning of LLaVA-1.5-7B on the alignment dataset.

Builds and runs the DeepSpeed command. LoRA rank 64, alpha 128, dropout 0.1,
learning rate 2e-5 on a cosine schedule, 20 epochs, CLIP ViT-L/14-336 vision
tower. Around 24 hours on a single RTX 4090, against roughly one hour to train
either CNN.

Needs LLAVA_REPO pointing at a checkout of haotian-liu/LLaVA, and a DeepSpeed
config at scripts/zero2.json inside it.

The 20 epochs here match the single-stage study of the single-stage study; the coarse-to-fine study
states 10 epochs at batch 4 for the two ViRA adapters, and no separate ViRA
training script survives.
"""

import os

# =====================================================
# User Configuration (edit before running)
# =====================================================

# Base LLaVA model checkpoint
MODEL_NAME = "/data/llava/llava-v1.5-7b"

# Dataset for fine-tuning (LLaVA JSON format)
FINETUNE_DATA_PATH = "/data/datasets/augmented_llava_dataset.json"

# Image directories referenced in FINETUNE_DATA_PATH
IMAGE_FOLDERS = ",".join([
    "/data/datasets/images/set1_augmented",
    "/data/datasets/images/set2_augmented"
])

DEEPSPEED_CONFIG = "./scripts/zero2.json"
OUTPUT_DIR = "./checkpoints/llava-v1.5-custom-lora"


# =====================================================
# Fine-tuning Command
# =====================================================

finetune_cmd = f"""
deepspeed ./llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.1 \
    --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_CONFIG} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {FINETUNE_DATA_PATH} \
    --image_folder "{IMAGE_FOLDERS}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir {OUTPUT_DIR} \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
"""


# =====================================================
# Execute
# =====================================================

if __name__ == "__main__":
    print("Starting LLaVA fine-tuning...")
    os.system(finetune_cmd)
    print("Fine-tuning launched. Monitor logs and checkpoints for progress.")
