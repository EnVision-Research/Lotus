
export CUDA=0

export BASE_TEST_DATA_DIR="datasets/eval/"

export CHECKPOINT_DIR="jingheya/lotus-depth-d-v2-0-disparity"
export OUTPUT_DIR="output/Depth_D_Eval"
export TASK_NAME="depth"

export MODE="regression"

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
        # The defualt `processing_res` is set in the configuration file of each dataset. 
