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



if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
        export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_dev/output"
        export DATA_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_dev/data"
else
        export DATA_DIR_BASE="../src/transformers/data/datasets"
        export OUTPUT_DIR_BASE="./output"
fi

echo "MACHINE_TO_RUN_ON=$MACHINE_TO_RUN_ON"
echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
echo "DATA_DIR_BASE=$DATA_DIR_BASE"
echo "EPOCHS=$EPOCHS"



export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="lex" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export SUB_TASK_TYPE="figerspecific" #options for TASK_SUB_TYPE (usually used only for delex)  include [oa, figerspecific, figerabstract, oass, simplener]
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE"

export TOY_DATA_DIR="toydata"
export TOY_DATA_DIR_PATH="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$TOY_DATA_DIR/"
export PYTHONPATH="../src"

#for laptop
#export PYTHONPATH="/Users/mordor/research/huggingface/src"
export BERT_MODEL_NAME="bert-base-cased" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR


echo "OUTPUT_DIR=$OUTPUT_DIR"



#get data only if its 1st epoch
if [ $EPOCHS = "1" ]; then
        echo "found epopch is equal to 1. going to download data"
        rm -rf $DATA_DIR
        ./get_fever_fnc_data.sh
        ./convert_to_mnli_format.sh
fi

echo "done with data download  TOY_DATA_DIR_PATH now is $TOY_DATA_DIR_PATH"

#create a small part of data as toy data. this will be used to run regresssion tests before the actual run starts
./reduce_size.sh  --data_path $TOY_DATA_DIR_PATH


#./run_glue.sh
./run_tests.sh

