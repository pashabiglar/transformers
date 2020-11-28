#!/usr/bin/env bash

#make sure ./get_fever_fnc_data.sh is run before this file

#cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#relative paths that worked on june 6th


echo "value of epochs in runglue.sh is $EPOCHS"

echo "TASK_TYPE is $TASK_TYPE"
if [ "$TASK_TYPE" = "mod2" ] ; then
 echo $DATA_DIR
 echo "task type is mod2"

fi

echo "TASK_TYPE is $TASK_TYPE"
if [ "$TASK_TYPE" = "mod1" ] ; then
 echo $DATA_DIR
 echo "task type is mod1"
fi

if [ "$TASK_TYPE" = "combined" ] ; then
 echo $DATA_DIR
 echo "task type is combined"
 args="$args --do_train_1student_1teacher"
fi

echo "data_dir is $DATA_DIR"


echo $args

mkdir -p OUTPUT_DIR

export EXAMPLES_DIR_PATH="../examples/text-classification/"
echo "value of MACHINE_TO_RUN_ON"
echo $MACHINE_TO_RUN_ON
env CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
       python3 ../examples/text-classification/load_trained_model_print_attention_weights.py -i $EXAMPLES_DIR_PATH -o $MACHINE_TO_RUN_ON
else
       #python3 ../examples/text-classification/load_trained_model_predict.py $args
       python ../examples/text-classification/load_trained_model_print_attention_weights.py -i $EXAMPLES_DIR_PATH -o $MACHINE_TO_RUN_ON
fi

