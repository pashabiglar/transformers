#!/usr/bin/env bash

if [ $# -gt 1 ]
then
   echo "inside reduce sixe. number of args is greater than 1"
   if [ $1 == "--data_path" ]; then
         export bert_format_base_folder_path="$2"
    fi
else
  echo "inside reduce size. number of args is not greater than 1"
  echo $#

fi


for complete_path in $(find $DATA_DIR -name '*.tsv');
do
if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
    head -17  $complete_path > temp
    mv temp $complete_path
else
    head -100  $complete_path > temp
    mv temp $complete_path
fi



done

