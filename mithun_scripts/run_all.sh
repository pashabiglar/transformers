#!/bin/bash

export MACHINE_TO_RUN_ON="laptop" #options include [laptop, hpc,clara]
export EPOCHS=1
if [ $# -gt 1 ]
then
   echo "number of args is greater than 1"
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
        export OUTPUT_DIR_BASE="output"
fi

echo $MACHINE_TO_RUN_ON
echo $OUTPUT_DIR_BASE
echo $DATA_DIR_BASE
echo $EPOCHS


export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="lex" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export SUB_TASK_TYPE="figerspecific" #options for TASK_SUB_TYPE (usually used only for delex)  include [oa, figerspecific, figerabstract, oass, simplener]
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE"
export PYTHONPATH="../src"
export BERT_MODEL_NAME="bert-base-uncased" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="512"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR


#commenting this on august 1st since downloading data was becoming a pain. due to tokenization issues. i.e after merging with
#latest code of HF, for some reason tokenization was taking 24+ hours. I decided to reuse the old cahced tokenizations instead of
#trying to figure out what happened due to merge. PIcking my battles.

#get data only if its 1st epoch
if [ $EPOCHS = "1" ]; then
        echo "found epopch is equal to 1. going to download data"
        rm -rf $DATA_DIR
        ./get_fever_fnc_data.sh
        ./convert_to_mnli_format.sh
fi

echo "done with data download part if epoch==1. datapath now is $DATA_DIR"




if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
      ./reduce_size.sh  --data_path $DATA_DIR
fi



./run_glue.sh

