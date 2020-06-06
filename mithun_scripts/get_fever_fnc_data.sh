#!/bin/bash


export BASE_DATA_DIR="../src/transformers/data/datasets/"

cd $BASE_DATA_DIR
rm -rf fever

#pick according to which kind of dataset you want to use for  train, dev, test on. Eg: train on fever, test on fnc

#######indomain fever

mkdir -p fever/feverindomain/lex


FILE=fever/feverindomain/lex/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/in-domain/lex/train.tsv -O $FILE
fi
