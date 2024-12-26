
export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-depth-g-v2-0-disparity"
export OUTPUT_DIR="output/Depth_G_Infer"
export TASK_NAME="depth"

# export CHECKPOINT_DIR="jingheya/lotus-normal-g-v1-0"
# export OUTPUT_DIR="output/Normal_G_Infer"
# export TASK_NAME="normal"

# export MODE="regression"
export MODE="generation"

export TEST_IMAGES="assets/in-the-wild_example"

CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --input_dir=$TEST_IMAGES \
        --task_name=$TASK_NAME \
        --mode=$MODE \
        --output_dir=$OUTPUT_DIR \
        --disparity 
        # --processing_res=0 # Defualt: 768. To obtain more fine-grained results, you can set `--processing_res=0` (original resolution) or a higher resolution. 