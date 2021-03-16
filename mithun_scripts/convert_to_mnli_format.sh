#!/usr/bin/env bash

for complete_path in $(find $DATA_DIR -name '*.tsv');
do
echo "going to convert the following file to mnli format"
echo $complete_path
python convert_fever_data_to_mnli_format.py --file_path $complete_path
done

