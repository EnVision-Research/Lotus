# export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_NAME="stabilityai/stable-diffusion-2-base"

# training dataset
export TRAIN_DATA_DIR_HYPERSIM=$PATH_TO_HYPERSIM_DATA
export TRAIN_DATA_DIR_VKITTI=$PATH_TO_VKITTI_DATA
export RES_HYPERSIM=576
export RES_VKITTI=375
export P_HYPERSIM=0.9

# training configs
export BATCH_SIZE=16
export CUDA=01234567
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs
export TIMESTEP=999
export TASK_NAME="normal"

# eval
export BASE_TEST_DATA_DIR="datasets/eval/"
export VALIDATION_IMAGES="datasets/quick_validation/"
export VAL_STEP=1000

# output dir
export OUTPUT_DIR="output/train-lotus-g-${TASK_NAME}-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  --main_process_port="13224" \
  train_lotus_g.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --train_data_dir_vkitti=$TRAIN_DATA_DIR_VKITTI \
  --resolution_vkitti=$RES_VKITTI \
  --prob_hypersim=$P_HYPERSIM \
  --mix_dataset \
  --random_flip \
  --align_cam_normal \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_images=$VALIDATION_IMAGES \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$VAL_STEP \
  --base_test_data_dir=$BASE_TEST_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="latest"