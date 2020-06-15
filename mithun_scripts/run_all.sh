#!/usr/bin/env bash
export DATA_DIR_BASE="../src/transformers/data/datasets"
export DATASET="fever"
export basedir="$DATA_DIR_BASE/$DATASET"
export TASK_TYPE="combined" #options for task type include lex,delex,and empty"". Empty is used in case of student teacher architecture which will load a paralleldataset from both lex and delex folders
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE"

#comment this section if you just downloaded and converted the data fresh using these.-useful for repeated runs
rm -rf $basedir
./get_fever_fnc_data.sh
# uncomment this if you want to run on a toy data
./reduce_size.sh
./convert_to_mnli_format.sh
#############end of commentable data sections
./run_glue.sh

