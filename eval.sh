
export CUDA=0

export BASE_TEST_DATA_DIR="datasets/eval/"

export CHECKPOINT_DIR="jingheya/lotus-depth-g-v2-0-disparity"
export OUTPUT_DIR="output/Depth_G_Eval"
export TASK_NAME="depth"

# export CHECKPOINT_DIR="jingheya/lotus-normal-g-v1-0"
# export OUTPUT_DIR="output/Normal_G_Eval"
# export TASK_NAME="normal"

# export MODE="regression"
export MODE="generation"

CUDA_VISIBLE_DEVICES=$CUDA python eval.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --base_test_data_dir=$BASE_TEST_DATA_DIR \
        --task_name=$TASK_NAME \
        --mode=$MODE \
        --output_dir=$OUTPUT_DIR \
        --disparity
