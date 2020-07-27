#!/usr/bin/env bash
#this can be used as a toy data for quick end to end testing
#todo: pick the correct data folder path based on the value of machine to run :server or laptop
#bert_format_base_folder_path="/xdisk/msurdeanu/mithunpaul/huggingface/transformers/src/transformers/data/datasets/fever/"
#cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
bert_format_base_folder_path="../src/transformers/data/datasets/fever"

for complete_path in $(find $bert_format_base_folder_path -name '*.tsv');
do
head -32  $complete_path > temp
mv temp $complete_path

done

