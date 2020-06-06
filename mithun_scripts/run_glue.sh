#!/usr/bin/env bash

#change these to corresponding locations
export PYTHONPATH="../src/"
export GLUE_DIR="../src/transformers/data/datasets/fever/feverindomain/lex/"
export TASK_NAME=feverindomain

python ../examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased     --task_name $TASK_NAME      --do_train     --do_eval      --data_dir $GLUE_DIR    --max_seq_length 128      --per_device_eval_batch_size=8        --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir /tmp/$TASK_NAME --overwrite_output_dir


# expanded run parameters for feverindomain
# --model_name_or_path bert-base-uncased     --task_name feverindomain      --do_train     --do_eval      --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/feverindomain/lex/      --max_seq_length 128      --per_device_eval_batch_size=8        --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir /tmp/feverindomain/ --overwrite_output_dir