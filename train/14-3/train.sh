#!/bin/bash

# Inference script with argument names only
python3 inference.py \
    --log_path \
    --batch_size \
    --learning_rate \
    --epoch \
    --train_file_path \
    --valid_file_path \
    --test_file_path \
    --save_path \
    --model_name \
    --hf_token \
    --num_feature \
    --load_in_4bit \
    --bnb_4bit_quant_type \
    --bnb_4bit_use_double_quant \
    --lora_rank \
    --lora_alpha \
    --target_modules \
    --lora_dropout \
    --bias
