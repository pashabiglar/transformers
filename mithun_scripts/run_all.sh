#!/usr/bin/env bash

export MACHINE_TO_RUN_ON="server" #options include [laptop, server]
export DATA_DIR_BASE="../src/transformers/data/datasets"
export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="delex" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export SUB_TASK_TYPE="figerspecific" #options for TASK_SUB_TYPE (usually used only for delex)  include [oa, figerspecific, figerabstract, oass, simplener]
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE"
export BERT_MODEL_NAME="bert-base-uncased" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
export OUTPUT_DIR_BASE="output"
if [ "$MACHINE_TO_RUN_ON" = "server" ]; then
    export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/output"
fi

export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$SUB_TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR


#export PYTHONPATH="/Users/mordor/research/transformers/src"
WANDB_API_KEY=de268c256c2d4acd9085ee4e05d91706c49090d7
#comment this section if you just downloaded and converted the data fresh using these.-useful for repeated runs
rm -rf $basedir

#for server clara
#torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

./get_fever_fnc_data.sh
./reduce_size.sh
./convert_to_mnli_format.sh
#############end of commentable data sections
./run_glue.sh

