#!/bin/bash

# Inference script with argument names only
python3 inference.py \
    --base_model_name \
    --trained_model_path \
    --device \
    --test_data_path \
    --token \
    --num_feature \
    --batch_size \
    --result_name \
    --log_name
