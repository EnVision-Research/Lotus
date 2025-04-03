#!/bin/bash

python utils/process_hypersim.py \
    --csv_path=$PATH_TO_HYPERSIM_SPLIT_CSV \
    --src_path=$PATH_TO_RAW_HYPERSIM_DATA \
    --trg_path=$PATH_TO_HYPERSIM_DATA \
    --split='train' \
    --filter_nan
