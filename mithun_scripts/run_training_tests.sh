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
echo $PYTHONPATH

if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
        cd ../examples/tests/
        pytest -s test_training_mithun_factverification.py



else
        cd ../examples/tests/
        pytest -s test_training_mithun_factverification.py


fi
