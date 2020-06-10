#!/usr/bin/env bash


cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#relative paths that worked on june 6th
export PYTHONPATH="../src/"
export GLUE_DIR="../src/transformers/data/datasets/fever/feverindomain/delex/"
export TASK_NAME=feverindomain
OUTPUT_DIR="output/"

mkdir -p output
#python /xdisk/msurdeanu/mithunpaul/huggingface/transformers/examples/text-classification/run_glue.py  --model_name_or_path bert-base-uncased     --task_name $TASK_NAME      --do_train     --do_eval   --do_predict   --data_dir $GLUE_DIR    --max_seq_length 128      --per_device_eval_batch_size=8        --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir $OUTPUT_DIR --overwrite_output_dir
python3.6 ../examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased     --task_name $TASK_NAME      --do_train   --do_eval      --data_dir $GLUE_DIR    --max_seq_length 128      --per_device_eval_batch_size=16        --per_device_train_batch_size=16       --learning_rate 1e-5      --num_train_epochs 5.0      --output_dir $OUTPUT_DIR --overwrite_output_dir  --save_steps 37000 --weight_decay 0.01 --adam_epsilon 1e-6


# expanded run parameters for feverindomain
# --model_name_or_path bert-base-uncased     --task_name feverindomain      --do_train     --do_eval      --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/feverindomain/lex/      --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6

#to load a trained model and do dev (--do_eval) or predict on test (--do_predict)
#absolute paths for laptop python3.6 ../examples/text-classification/run_glue.py --model_name_or_path "/Users/mordor/research/huggingface/trained_models/checkpoint-37000/" --tokenizer_name  bert-base-uncased    --task_name fevercrossdomain   --do_predict     --data_dir "/Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/lex"     --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6
#relative paths for server
#python3.6 ../examples/text-classification/run_glue.py --model_name_or_path "output/checkpoint-37000/" --tokenizer_name  bert-base-uncased    --task_name fevercrossdomain   --do_predict     --data_dir "../src/transformers/data/datasets/fever/fevercrossdomain/lex"     --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6

