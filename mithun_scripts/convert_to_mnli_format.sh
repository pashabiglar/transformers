#!/usr/bin/env bash

cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
#bert_format_base_folder_path="/xdisk/msurdeanu/mithunpaul/huggingface/transformers/src/transformers/data/datasets/fever/"
bert_format_base_folder_path="../src/transformers/data/datasets/fever"

for complete_path in $(find $bert_format_base_folder_path -name '*.tsv');
do
echo "going to convert the following file to mnli format"
echo $complete_path
python convert_fever_data_to_mnli_format.py --file_path $complete_path
done

