#!/bin/bash





#
#
########fevercrossdomain lex (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
#if [ "$TASK_TYPE" = "lex" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then
#
#echo "found task type is lex and task name as fever cross domain"
#
#
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#
#
#FILE=$DATA_DIR/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://osf.io/r6mdz/download -O $FILE
#fi
#
#FILE=$DATA_DIR/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://osf.io/azf6t/download -O $FILE
#fi
#
#
#
##note that the test file is fnc dev partition
#FILE=$DATA_DIR/test.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#
#        # fnc-test lexicalized/plaintext
#      wget https://osf.io/r5uvd/download -O $FILE
#
#fi
#fi
#########fevercrossdomain delex where delexicalization was done using overlap aware (oa) technique (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
#if [ "$TASK_TYPE" = "delex" ] && [ "$TASK_NAME" = "fevercrossdomain" ]  && [ "$SUB_TASK_TYPE" = "oa" ]; then
#echo "found task type is lex and task name as fever cross domain and SUB_TASK_TYPE is oa"
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#
#
#FILE=$DATA_DIR/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/train.tsv -O $FILE
#fi
#
#FILE=$DATA_DIR/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/dev.tsv -O $FILE
#fi
#
##note that we are already replacing the file to be tested as test file. this way during run time you have to just
## load all files as is e.g #if do_predict is true load from folder fevercrossdomain/delex/test.tsv- which the code already does.
#
#FILE=$DATA_DIR/test.tsv
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/oa/dev.tsv -O $FILE
#fi
#fi
#
#
#########fevercrossdomain delex where delexicalixation was done with figer-specific technique (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
#if [ "$TASK_TYPE" = "delex" ] && [ "$TASK_NAME" = "fevercrossdomain" ]  && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then
#
#echo "found task type is delex and task name as fever cross domain and sub task type ==figerspecific"
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#
#FILE=$DATA_DIR/train.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer_specific/train.tsv -O $FILE
#fi
#
#FILE=$DATA_DIR/dev.tsv
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#   wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer_specific/dev.tsv -O $FILE
#fi
#
##We are loading the cross domain dev file (eg: fnc-dev file) as the test partition
## This is just a hack so that the code produces/predicts results in single go, as opposed to having to reload a trained model
#FILE=$DATA_DIR/test.tsv
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/figer_specific/dev.tsv -O $FILE
#fi
#fi
#
#
#####################################for cross domain student teacher, there will be two training files.-one for lex and another for delex
#
#
#if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then
#    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be figerspecific"
#
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#FILE="$DATA_DIR/train1.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#
#    wget https://osf.io/r6mdz/download -O $FILE
#fi
#
#
#FILE="$DATA_DIR/train2.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#      wget https://osf.io/8shu4/download -O $FILE
#fi
#
#FILE="$DATA_DIR/dev.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#
#    #uncomment this if you want to feed lexicalized version of the dataset (fever-dev) as  dev partition. this is useful when you want to sanity check  how a lexicalized model is performing
#    #wget https://osf.io/azf6t/download -O $FILE
#
#    wget https://osf.io/r5pz3/download -O $FILE
#
#fi
#
#
##note that we are  replacing the test partition with cross domain dev partition(in this case. it thus becomes the in-domain dev partition of fnc dataset).
#
#FILE="$DATA_DIR/test.tsv"
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#
#      # if you want to use the lexicalized version of the dataset (fnc-dev) as the test partition.
#      # this is useful when you want to sanity check  how a lexicalized model is performing
#
#      #fnc-dev lexicalized plaintext
#      #wget https://osf.io/qs4u6/download -O $FILE
#
#      # fnc-dev delexicalized using figerspecific
#      #wget https://osf.io/jx32m/download   -O $FILE
#
#      # fnc-test delexicalized using figerspecific
#      wget https://osf.io/jentp/download   -O $FILE
#
#
#        # fnc-test lexicalized/plaintext
#      #wget https://osf.io/r5uvd/download -O $FILE
#
#
#
#
#fi
#
#fi
#
#
#####################################for cross domain student teacher, there will be two training files.-one for mod1 and another for mod2
#
#
#if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerabstract" ]; then
#    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be figerabstract"
#
##note, train1.tsv will be the lexicalized version of the file(so the link below almost always never changes) while train2.tsv is the delexicalized, which can change
##based on the type of delexicalization algorithm used. eg:figerabstract
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#FILE="$DATA_DIR/train1.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/lex/train.tsv     -O $FILE
#fi
#
#
#FILE="$DATA_DIR/train2.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#      wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer-abstract/train.tsv -O $FILE
#fi
#
#FILE="$DATA_DIR/dev.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer-abstract/dev.tsv -O $FILE
#
#fi
#
#
##note that we are  replacing the test partition with cross domain dev partition(in this case it thus becomes the in-domain dev
## partition of fnc dataset).
#
#FILE="$DATA_DIR/test.tsv"
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#
#      #uncomment this if you want to feed dev and test as lex. this is used when you want to check if lex model alone works fine from within student teacher
#      #wget https://storage.googleapis.com/fact_ve®©rification_mithun_files/TSV/FNC/in-domain/lex/dev.tsv -O $FILE
#      wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/figer_specific/dev.tsv   -O $FILE
#
#
#
#fi
#
#fi
#
#
#
#####################################for cross domain student teacher, when delex files are delexicalized with oa
#
#
#
#
#
#if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "oa" ]; then
#    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be oa"
#
#echo $DATA_DIR
#mkdir -p $DATA_DIR
#
#FILE="$DATA_DIR/train1.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/lex/train.tsv     -O $FILE
#fi
#
#
#FILE="$DATA_DIR/train2.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/train.tsv -O $FILE
#fi
#
#FILE="$DATA_DIR/dev.tsv"
#if test -f "$FILE";then
#    echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/dev.tsv -O $FILE
#fi
#
##note that we are already replacing the file to be tested as test file. this way during run time you have to just
## load all files as is e.g #if do_predict is true load from folder fevercrossdomain/delex/test.tsv- which the code already does.
#
#FILE="$DATA_DIR/test.tsv"
#if test -f "$FILE";then
#echo "$FILE exists"
#else
#    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/oa/dev.tsv -O $FILE
#fi
#
#fi
#
#
#
#####################################for testing with a trained lex model


if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE1" = "lex" ] && [ "$SUB_TASK_TYPE2" = "lex" ]; then
    echo "found task type to be 3t1s, taskname to be feverCrossDomain and subtasktypes to be lex and lex"


echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training with lexicalized/plain text version of fever
     wget https://osf.io/r6mdz/download  -O $FILE
fi



#in the middle of loading a model and testing with it.
# for fever2fnc we will load dev as the corresponding fnc-dev- (indomain aka 4 labels) /itself. that way we can confirm on the fly that we have the right model
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

    #fever-test indomain figerspec to get indomain test value for paper
    #wget https://osf.io/xycv9/download -O $FILE

    #fever-test indomain lex to get indomain test value for paper

    #fnc-dev-lex aka plaintext (indomain aka 4 labels)
    # wget https://osf.io/jfpbv/download -O $FILE

    #fnc-dev-delex by figerspecific(indomain aka 4 labels)
    # wget https://osf.io/jx32m/download -O $FILE

    #plain text version of fever-dev partition with 3labels. this is to calclulate for acl paper
    #wget https://osf.io/xdbh6/download -O $FILE

    wget https://osf.io/azf6t/download -O #fever_dev_lex_4labels.txt
fi




FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
#plain text version of fnc-test partition with 4labels
     #wget https://osf.io/r5uvd/download -O $FILE

#delex figerspecific version of fnc-test partition with 4labels
     #wget https://osf.io/jentp/download  -O $FILE

    #plain text version of fever-test partition with 4labels. this is to calclulate for acl paper
    #wget https://osf.io/85h4z/download -O $FILE

     #plain text version of fever-test partition with 3labels. this is to calclulate for acl paper
     #wget https://osf.io/q38pn/download -O $FILE

     #plain text version of fever-test partition with 4labels. this is to calclulate for acl paper
    wget https://osf.io/85h4z/download -O $FILE

fi

#fi of if corresponding to 3t1s check
fi




####################################for testing with fnccrossdomain


if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fnccrossdomain" ] && [ "$SUB_TASK_TYPE1" = "lex" ] && [ "$SUB_TASK_TYPE2" = "lex" ]; then
    echo "found task type to be 3t1s, taskname to be feverCrossDomain and subtasktypes to be lex and lex"


echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training with lexicalized/plain text version of fever
     wget https://osf.io/r6mdz/download  -O $FILE
fi



#in the middle of loading a model and testing with it.
# for fever2fnc we will load dev as the corresponding fnc-dev- (indomain aka 4 labels) /itself. that way we can confirm on the fly that we have the right model
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

    #fever-test indomain figerspec to get indomain test value for paper
    #wget https://osf.io/xycv9/download -O $FILE

    #fever-test indomain lex to get indomain test value for paper

    #fnc-dev-lex aka plaintext (indomain aka 4 labels)
    #wget https://osf.io/jfpbv/download -O $FILE

    #fnc-dev-delex by figerspecific(indomain aka 4 labels)
     wget https://osf.io/jx32m/download -O $FILE

    #plain text version of fever-dev partition with 3labels. this is to calclulate for acl paper
    #wget https://osf.io/xdbh6/download -O $FILE
fi




FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
#plain text version of fnc-test partition with 4labels
     #wget https://osf.io/r5uvd/download -O $FILE

#delex figerspecific version of fnc-test partition with 4labels
     #wget https://osf.io/jentp/download  -O $FILE

    #plain text version of fever-test partition with 4labels. this is to calclulate for acl paper
    wget https://osf.io/85h4z/download -O $FILE

     #plain text version of fever-test partition with 3labels. this is to calclulate for acl paper
     #wget https://osf.io/q38pn/download -O $FILE

#will load fnc-teest plain text with 3 labels(aka crossdomain)
#wget https://osf.io/64syf/download -O $FILE

#will load fnc-teest oaner with 3 labels(aka crossdomain)
#wget https://osf.io/eywd2/download -O $FILE

fi

#fi of if corresponding to 3t1s check
fi