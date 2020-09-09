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

echo "inside reduce size.value of directory to reduce files from is"
echo $bert_format_base_folder_path

for complete_path in $(find $bert_format_base_folder_path -name '*.tsv');
do


if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
    head -9000  $complete_path > temp
    mv temp $complete_path
else
    head -100  $complete_path > temp
    mv temp $complete_path
fi


done

