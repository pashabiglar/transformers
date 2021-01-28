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
        export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fever_to_fnc_run_training_4models_classweight0.1/output"
        export DATA_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fever_to_fnc_run_training_4models_classweight0.1/data"
fi

if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
        wandb off
        export DATA_DIR_BASE="/Users/mordor/research/huggingface/src/transformers/data/datasets"
        export OUTPUT_DIR_BASE="/Users/mordor/research/huggingface/mithun_scripts/output"
        export PYTHONPATH="/Users/mordor/research/huggingface/src/"
fi

if [ $MACHINE_TO_RUN_ON == "clara" ]; then
        export OUTPUT_DIR_BASE="/work/mithunpaul/huggingface_bertmini_lex_standalone/output"
        export DATA_DIR_BASE="/work/mithunpaul/huggingface_bertmini_lex_standalone/data"
fi


echo "MACHINE_TO_RUN_ON=$MACHINE_TO_RUN_ON"
echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
echo "DATA_DIR_BASE=$DATA_DIR_BASE"
echo "EPOCHS=$EPOCHS"



export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="combined" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export SUB_TASK_TYPE="figerspecific" #options for TASK_SUB_TYPE (usually used only for TASK_TYPEs :[delex,combined])  include [oa, figerspecific, figerabstract, oass, simplener]
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE"
export PYTHONPATH="../src"
export BERT_MODEL_NAME="google/bert_uncased_L-12_H-128_A-2" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR
wandb on
wandb online

echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "value of epochs is $EPOCHS"
echo "value of DATA_DIR is $DATA_DIR"



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

echo "done with data download part . datapath now is $DATA_DIR"



if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
      ./reduce_size.sh  --data_path $DATA_DIR
fi



./run_glue.sh

