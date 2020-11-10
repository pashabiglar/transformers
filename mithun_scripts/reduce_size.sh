#!/usr/bin/env bash

if [ $# -gt 1 ]
then
   echo "inside reduce sixe. number of args is greater than 1"
   if [ $1 == "--data_path" ]; then
         export toy_data_path="$2"
    fi
else
  echo "inside reduce_size,sh. number of args is not greater than 1. going to exit"
  echo $#
  exit
fi

#create a data directory where the reduced size toy data will be kept. Initially full data is downloaded here , then later when reduce_size.sh is called, the data is reduced
mkdir -p $toy_data_path
echo "value of toy_data_path is $toy_data_path"


for complete_path in $(find $DATA_DIR -name '*.tsv');
do

FILE="`basename $complete_path`"
echo $FILE
#laptop cant handle more than 17 data points at a time
if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
    head -35  $complete_path > temp
    toy_data_full_path="$toy_data_path$FILE"
    mv temp $toy_data_full_path
    echo "found that MACHINE_TO_RUN_ON is laptop . reduced size toy data created at $toy_data_full_path"
else
    head -100  $complete_path > temp
    toy_data_full_path="$toy_data_path$FILE"
    mv temp $toy_data_full_path
    echo "found that MACHINE_TO_RUN_ON is laptop . reduced size toy data created at to $toy_data_full_path"
fi





done

