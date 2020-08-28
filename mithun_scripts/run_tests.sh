#!/usr/bin/env bash

#make sure ./get_fever_fnc_data.sh is run before this file

#cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#relative paths that worked on june 6th


echo "value of epochs in runglue.sh is $EPOCHS"


args="--model_name_or_path $BERT_MODEL_NAME   --task_name $TASK_NAME      --do_train   --do_eval   --do_predict    \
--data_dir $DATA_DIR    --max_seq_length $MAX_SEQ_LENGTH      --per_device_eval_batch_size=16        --per_device_train_batch_size=16       \
--learning_rate 1e-5      --num_train_epochs $EPOCHS     --output_dir $OUTPUT_DIR --overwrite_output_dir  \
--weight_decay 0.01 --adam_epsilon 1e-6  --evaluate_during_training \
--task_type $TASK_TYPE --subtask_type $SUB_TASK_TYPE --machine_to_run_on $MACHINE_TO_RUN_ON"

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

if [ "$TASK_TYPE" = "combined" ] ; then
 echo $DATA_DIR
 echo "task type is combined"
 args="$args --do_train_1student_1teacher"
fi

echo "data_dir is $DATA_DIR"


echo $args



mkdir -p OUTPUT_DIR


if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
        cd ../examples/
        python3.6 -m unittest test_examples_mithun_factverification.py $args
else
        cd ../examples/
        #python test_examples_mithun_factverification.py  $args
        pytest test_examples_mithun_factverification.py

fi
