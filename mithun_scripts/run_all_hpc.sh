#!/usr/bin/env bash
cd /home/u11/mithunpaul/xdisk/huggingface/transformers/mithun_scripts
bash get_fever_fnc_data.sh

# uncomment this if you want to run on a toy data
bash reduce_size.sh

bash convert_to_mnli_format.sh
bash run_glue.sh

