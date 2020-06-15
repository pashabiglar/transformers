#!/bin/bash










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


#######fevercrossdomain lex (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)

#mkdir -p fever/fevercrossdomain/lex
#
#FILE=fever/fevercrossdomain/lex/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/lex/train.tsv     -O $FILE
#fi
#
#FILE=fever/fevercrossdomain/lex/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/FEVER/cross-domain/lex/dev.tsv -O $FILE
#fi
#
##note that the test file is fnc dev partition
#FILE=fever/fevercrossdomain/lex/test.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/lex/dev.tsv -O $FILE
#fi
#
########fevercrossdomain delex (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
#
#mkdir -p fever/fevercrossdomain/delex
#
#FILE=fever/fevercrossdomain/delex/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/train.tsv -O $FILE
#fi
#
#FILE=fever/fevercrossdomain/delex/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/dev.tsv -O $FILE
#fi
#
##note that we are already replacing the file to be tested as test file. this way during run time you have to just
## load all files as is e.g #if do_predict is true load from folder fevercrossdomain/delex/test.tsv- which the code already does.
#
#FILE=fever/fevercrossdomain/delex/test.tsv
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/oa/dev.tsv -O $FILE
#fi

####################################for cross domain student teacher, there will be two training files.-one for lex and another for delex


if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] ; then
    echo "found task type to be combined"

echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/lex/train.tsv     -O $FILE
fi


FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/train.tsv -O $FILE
fi

FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/dev.tsv -O $FILE
fi

#note that we are already replacing the file to be tested as test file. this way during run time you have to just
# load all files as is e.g #if do_predict is true load from folder fevercrossdomain/delex/test.tsv- which the code already does.

FILE="$DATA_DIR/test.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/oa/dev.tsv -O $FILE
fi

fi