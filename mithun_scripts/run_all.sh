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
        export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/output"
        export DATA_DIR_BASE="/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/data"
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
export BERT_MODEL_NAME="bert-base-cased" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/$EPOCHS/"
echo $OUTPUT_DIR


#get data only if its 1st epoch
if [ $EPOCHS = "1" ]; then
        echo "found epopch is equal to 1. going to download data"
        rm -rf $DATA_DIR
        ./get_fever_fnc_data.sh
        ./convert_to_mnli_format.sh
fi


./reduce_size.sh

if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
      ./reduce_size.sh
fi



./run_glue.sh

