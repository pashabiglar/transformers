#!/bin/bash



export MACHINE_TO_RUN_ON="laptop" #options include [laptop, hpc,clara]
export EPOCHS=1
if [ $# -gt 1 ]
then
   echo "number of args is $#"
   if [ $1 == "--epochs_to_run" ]; then
         export EPOCHS="$2"
    fi
else
  echo "number of args is not greater than 1"
  export EPOCHS=1
fi





if [ $# -gt 2 ]; then
if [ $3 == "--machine_to_run_on" ]; then
        export MACHINE_TO_RUN_ON=$4
fi
fi



if [ $# -gt 3 ]; then
if [ $5 == "--use_toy_data" ]; then
        export USE_TOY_DATA=$6
fi
fi


if [ $# -gt 7 ]; then
if [ $7 == "--download_fresh_data" ]; then
        export DOWNLOAD_FRESH_DATA=$8
fi
fi



if [ $MACHINE_TO_RUN_ON == "hpc" ]; then


        export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fever_to_fnc_run_training/output"
        export DATA_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fever_to_fnc_run_training/data"
else
        wandb off
        export DATA_DIR_BASE="/Users/mordor/research/huggingface/src/transformers/data/datasets"
        export OUTPUT_DIR_BASE="/Users/mordor/research/huggingface/mithun_scripts/output"
        export PYTHONPATH="/Users/mordor/research/huggingface/src/"

fi

echo "MACHINE_TO_RUN_ON=$MACHINE_TO_RUN_ON"
echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
echo "DATA_DIR_BASE=$DATA_DIR_BASE"
echo "EPOCHS=$EPOCHS"



export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="2t1s" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both mod1 and mod2 folders
export SUB_TASK_TYPE1="figerspecific" #options for TASK_SUB_TYPE (usually used only for TASK_TYPEs :[mod2,combined])  include [oa, figerspecific, figerabstract, oass, simplener]
export SUB_TASK_TYPE2="oa" #options for TASK_SUB_TYPE (usually used only for TASK_TYPEs :[mod2,combined])  include [oa, figerspecific, figerabstract, oass, simplener]
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE1"

export TOY_DATA_DIR="toydata"
export TOY_DATA_DIR_PATH="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE1/$TOY_DATA_DIR/"



export BERT_MODEL_NAME="bert-base-cased" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR

echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "value of epochs is $EPOCHS"
echo "value of DATA_DIR is $DATA_DIR"



#get data fresh before every run
echo ". going to download data"




echo "$DOWNLOAD_FRESH_DATA"
if [ $DOWNLOAD_FRESH_DATA == "true" ]; then
    echo "found DOWNLOAD_FRESH_DATA is true "
    rm -rf $DATA_DIR
    ./get_fever_fnc_data.sh
    ./convert_to_mnli_format.sh
fi

#create a small part of data as toy data. this will be used to run regresssion tests before the actual run starts
./reduce_size.sh  --data_path $TOY_DATA_DIR_PATH

echo "done with data download  TOY_DATA_DIR_PATH now is $TOY_DATA_DIR_PATH"


#use a smaller toy data to test

if  [ "$USE_TOY_DATA" = true ]; then
        DATA_DIR=$TOY_DATA_DIR_PATH
        echo "found USE_TOY_DATA is true"
fi




export args="--model_name_or_path $BERT_MODEL_NAME   --task_name $TASK_NAME      --do_train   --do_eval   --do_predict    \
--data_dir $DATA_DIR    --max_seq_length $MAX_SEQ_LENGTH      --per_device_eval_batch_size=16        --per_device_train_batch_size=16       \
--learning_rate 1e-5      --num_train_epochs $EPOCHS     --output_dir $OUTPUT_DIR --overwrite_output_dir  \
--weight_decay 0.01 --adam_epsilon 1e-6  --evaluate_during_training \
--task_type $TASK_TYPE --subtask_type1 $SUB_TASK_TYPE1 --subtask_type2 $SUB_TASK_TYPE2 --machine_to_run_on $MACHINE_TO_RUN_ON --toy_data_dir_path $TOY_DATA_DIR_PATH "



#test cases
#./run_training_tests.sh
#./run_loading_tests.sh



#actual code runs
./run_glue.sh

#./load_model_test.sh


