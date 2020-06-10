#!/bin/bash


#line which worked from june 6th noon
export BASE_DATA_DIR="../src/transformers/data/datasets/"

cd $BASE_DATA_DIR
#comment this folder removal only if you are sure that the data you have is in fever tsv format. else its better to remove and download data fresh.
rm -rf fever

#pick according to which kind of dataset you want to use for  train, dev, test on. Eg: train on fever, test on fnc

#######indomain fever lex

#mkdir -p fever/feverindomain/lex
#
#FILE=fever/feverindomain/lex/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/in-domain/lex/train.tsv -O $FILE
#fi
#
#FILE=fever/feverindomain/lex/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/in-domain/lex/dev.tsv -O $FILE
#fi


##############in domain fever delex oaner
#mkdir -p fever/feverindomain/delex
#
#FILE=fever/feverindomain/delex/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/in-domain/oa/train.tsv -O $FILE
#fi
#
#FILE=fever/feverindomain/delex/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/in-domain/oa/dev.tsv -O $FILE
#fi


#######fevercrossdomain (trainign and dev will be in fever (with 4 labels), and test on fnc-dev partition)

mkdir -p fever/fevercrossdomain/lex

FILE=fever/fevercrossdomain/lex/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/lex/train.tsv     -O $FILE
fi
#note that the dev file is fnc dev partition
FILE=fever/fevercrossdomain/lex/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/lex/dev.tsv -O $FILE
fi


#note that the test file is fnc test partition
#update @june 10th 2020. will predict on dev to calculate fnc-score
FILE=fever/fevercrossdomain/lex/test.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/lex/dev.tsv -O $FILE
fi

