#!/usr/bin/env bash
export DATA_DIR_BASE="../src/transformers/data/datasets"
export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE = "combined" #options for task type include lex,delex,and empty"". Empty is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
rm -rf $basedir #comment this folder removal only if you are sure that the data you have is in fever tsv format.its better download data fresh.
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE"

./get_fever_fnc_data.sh

# uncomment this if you want to run on a toy data
#./reduce_size.sh

./convert_to_mnli_format.sh
./run_glue.sh

