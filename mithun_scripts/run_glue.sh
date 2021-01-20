#!/usr/bin/env bash

#make sure ./get_fever_fnc_data.sh is run before this file

#cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#relative paths that worked on june 6th


echo "value of epochs in runglue.sh is $EPOCHS"

echo "TASK_TYPE is $TASK_TYPE"
if [ "$TASK_TYPE" = "delex" ] ; then
 echo $DATA_DIR
 echo "task type is delex"

fi

echo "TASK_TYPE is $TASK_TYPE"
if [ "$TASK_TYPE" = "lex" ] ; then
 echo $DATA_DIR
 echo "task type is lex"
fi


if [ "$TASK_TYPE" = "2t1s" ] ; then
 echo $DATA_DIR
 echo "task type is combined"
 args="$args --do_train_student_teacher"
fi


echo "TASK_TYPE is $TASK_TYPE"


echo $args


mkdir -p OUTPUT_DIR

env CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
       python3.6 ../examples/text-classification/run_glue.py $args
else
       python3 ../examples/text-classification/run_glue.py $args
fi



# for pycharm run->edit_configurations
#when adding new remember to make all paths absolute, not relative
#feverindomain
# --model_name_or_path bert-base-uncased     --task_name feverindomain      --do_train     --do_eval --do_predict      --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/feverindomain/lex/      --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6

#fever cross domain with student teacher
##--model_name_or_path bert-base-cased --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/combined/figerspecific --max_seq_length 128 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1 --output_dir output/fever/fevercrossdomain/combined/figerspecific/bert-base-cased/128/ --overwrite_output_dir --weight_decay 0.01 --adam_epsilon 1e-6 --evaluate_during_training --task_type combined --subtask_type figerspecific --do_train_1student_1teacher

#parameters when in student teacher with bert uncased
#--model_name_or_path bert-base-uncased --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/combined/figerspecific --max_seq_length 128 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1 --output_dir output/fever/fevercrossdomain/combined/figerspecific/bert-base-uncased/128/ --overwrite_output_dir --weight_decay 0.01 --adam_epsilon 1e-6 --evaluate_during_training --task_type combined --subtask_type figerspecific --overwrite_cache --do_train_1student_1teacher

#fever ceoss domain delex
#--model_name_or_path bert-base-uncased --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/transformers/src/transformers/data/datasets/fever/fevercrossdomain/delex --max_seq_length 128 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1.0 --output_dir output/ --overwrite_output_dir --save_steps 74000 --weight_decay 0.01 --adam_epsilon 1e-6 --overwrite_cache

#to load a trained model
#python3.6 ../examples/text-classification/run_glue.py --model_name_or_path "output/checkpoint-37000/" --tokenizer_name  bert-base-uncased    --task_name fevercrossdomain   --do_predict     --data_dir "../src/transformers/data/datasets/fever/fevercrossdomain/lex"     --max_seq_length 128      --per_device_eval_batch_size=8       --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 1.0      --output_dir "./output" --overwrite_output_dir --save_steps 14000 --weight_decay 0.01 --adam_epsilon 1e-6

#cross domain
#--model_name_or_path bert-base-uncased --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/lex/figerspecific --max_seq_length 512 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1 --output_dir output/fever/fevercrossdomain/lex/figerspecific/bert-base-uncased/512/ --overwrite_output_dir --weight_decay 0.01 --adam_epsilon 1e-6 --evaluate_during_training --task_type lex --subtask_type figerspecific --overwrite_cache

#cross domain delex bert cased
#--model_name_or_path bert-base-cased --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/transformers/src/transformers/data/datasets/fever/fevercrossdomain/delex/figerspecific --max_seq_length 128 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1.0 --output_dir output/ --overwrite_output_dir --save_steps 74000 --weight_decay 0.01 --adam_epsilon 1e-6 --overwrite_cache
