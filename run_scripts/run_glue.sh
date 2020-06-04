#!/usr/bin/env bash
export GLUE_DIR="/Users/mordor/research/huggingface/src/transformers/data/datasets/"
export TASK_NAME=QNLI

python ../examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name MNLI \
    --do_train \
    --do_eval \
    --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/MNLI/ \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/MNLI/


#version for edit configurations in pycharm
# --model_name_or_path bert-base-uncased     --task_name QNLI      --do_train     --do_eval      --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/QNLI/      --max_seq_length 128      --per_device_eval_batch_size=8        --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir /tmp/QNLI/