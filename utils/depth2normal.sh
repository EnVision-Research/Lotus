#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD

CUDA_VISIBLE_DEVICES=1 python utils/depth2normal.py \
    --data_path $PATH_TO_VKITTI_DATA \
    --batch_size 10 \
    --scenes 01 02 06 18 20