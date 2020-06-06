#!/usr/bin/env bash
./get_fever_fnc_data.sh
./convert_to_mnli_format.sh
./run_glue.sh

