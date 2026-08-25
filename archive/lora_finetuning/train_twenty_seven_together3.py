import os

MODEL_NAME = LLAVA_BASE # Change to 1.6 if needed

# Dataset Paths (Pre-Split Train & Validation Datasets)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "camera1_crop2_coarse_fine2_merged.json")

IMAGE_FOLDERS = ",".join([

    os.path.join(BASE_DIR, "opentron_station31/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station32/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station21/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station22/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station23/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station11/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station12/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station13/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station14/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station15/color_images/camera1_renamed_cropped2"),
    os.path.join(BASE_DIR, "opentron_station16/color_images/camera1_renamed_cropped2"),
])


#!/bin/bash
finetune_script = f'''
deepspeed {LLAVA_REPO}/llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.1 --mm_projector_lr 2e-5 \
    --deepspeed {LLAVA_REPO}/scripts/zero2.json \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {TRAIN_DATA_PATH} \
    --image_folder  "{IMAGE_FOLDERS}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-opentron-camera1-crop2-coarse-fine-lora \
    --num_train_epochs 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
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
'''
# b24331a193ddfda8f488353353da14458b3b7a49
# Run fine-tuning
os.system(finetune_script)
