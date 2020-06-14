#!/usr/bin/env bash


cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#relative paths that worked on june 6th
export PYTHONPATH="../src/"

#laptop
export PYTHONPATH="/Users/mordor/research/huggingface"



export DATA_DIR_BASE="../src/transformers/data/datasets/fever/"
export TASK_NAME="fevercrossdomain/"

export DATA_DIR="$DATA_DIR_BASE$TASK_NAME$TASK_TYPE"



if test $TASK_TYPE=="lex/"; then
 echo $DATA_DIR
 echo "task type is lex"
 exit
fi


OUTPUT_DIR="output/"

mkdir -p output
python3.6 ../examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased     --task_name $TASK_NAME      --do_train   --do_eval --do_train_1student_1teacher  --do_predict    --data_dir $DATA_DIR    --max_seq_length 128      --per_device_eval_batch_size=16        --per_device_train_batch_size=16       --learning_rate 1e-5      --num_train_epochs 5.0      --output_dir $OUTPUT_DIR --overwrite_output_dir  --save_steps 37000 --weight_decay 0.01 --adam_epsilon 1e-6


# hardcoded run parameters to be used for running from pycharm
#feverindomain
# --model_name_or_path bert-base-uncased     --task_name feverindomain      --do_train     --do_eval --do_predict      --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/feverindomain/lex/      --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6
#fever cross domain with student teacher
# --model_name_or_path bert-base-uncased     --task_name fevercrossdomain      --do_train     --do_eval   --do_predict     --data_dir "/Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain"      --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6
#--do_train_1student_1teacher
#to load a trained model and do dev (--do_eval) or predict on test (--do_predict)
#absolute paths for laptop python3.6 ../examples/text-classification/run_glue.py --model_name_or_path "/Users/mordor/research/huggingface/trained_models/checkpoint-37000/" --tokenizer_name  bert-base-uncased    --task_name fevercrossdomain   --do_predict     --data_dir "/Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/lex"     --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6
#relative paths for server
#python3.6 ../examples/text-classification/run_glue.py --model_name_or_path "output/checkpoint-37000/" --tokenizer_name  bert-base-uncased    --task_name fevercrossdomain   --do_predict     --data_dir "../src/transformers/data/datasets/fever/fevercrossdomain/lex"     --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6

