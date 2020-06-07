#!/usr/bin/env bash
./get_fever_fnc_data.sh

# uncomment this if you want to run on a toy data
./reduce_size.sh

./convert_to_mnli_format.sh
./run_glue.sh

