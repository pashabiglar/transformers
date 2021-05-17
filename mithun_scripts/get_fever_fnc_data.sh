#!/bin/bash





#########################################################################################################################################################################################
# fevercrossdomain with one model that uses lexicalized dataset (i.e training and dev will be in fever
# (with 4 labels), and test on fnc-dev partition)
# #+ few shot== a few data points from cross domain target dataset is added to the end of training data
if [ "$TASK_TYPE" = "lex" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUBTASK_TYPE" = "few_shot" ]; then

echo "found task type is lex and task name as fever cross domain and SUBTASK_TYPE is few_shot"

echo $DATA_DIR
mkdir -p $DATA_DIR



FILE=$DATA_DIR/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/r6mdz/download -O $FILE
    # for few shot learning
    # here we download lex plain text version of fnc in domain with 4 labels and append it at the end of fever4 labels in domain training data
    wget https://osf.io/a6tks/download -O cross_domain_train.tsv
    tail -300 cross_domain_train.tsv >> $FILE
fi

FILE=$DATA_DIR/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/azf6t/download -O $FILE
fi



#note that the test file is fnc dev partition
FILE=$DATA_DIR/test.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
      #fnc-dev-lexicalized/
      wget https://osf.io/jfpbv//download -O $FILE

      #fnc-test lexicalized/plaintext- use this only once for final testing
      #wget https://osf.io/r5uvd/download -O $FILE


fi
fi


#########################################################################################################################################################################################

# fevercrossdomain with one model that uses lexicalized dataset (i.e training and dev will be in fever
# (with 4 labels), and test on fnc-dev partition)
if [ "$TASK_TYPE" = "lex" ] && [ "$TASK_NAME" = "fevercrossdomain" ] ; then

echo "found task type is lex and task name as fever cross domainxxxx"



echo $DATA_DIR
mkdir -p $DATA_DIR


FILE=$DATA_DIR/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/r6mdz/download -O $FILE
fi


FILE=$DATA_DIR/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    #wget https://osf.io/azf6t/download -O $FILE

    #in-domain test partition. to be used only once for paper
    wget https://osf.io/85h4z/download -O $FILE


fi



#note that the test partition file usually is cross domain(in this case fnc) dev partition
FILE=$DATA_DIR/test.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
      #fnc-dev-lexicalized/
      #wget https://osf.io/jfpbv//download -O $FILE

      #fnc-test lexicalized/plaintext- use this only once for final testing
      wget https://osf.io/r5uvd/download -O $FILE

fi
fi



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
    wget https://osf.io/r6mdz/download -O $FILE
fi

FILE=$DATA_DIR/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/azf6t/download -O $FILE
fi



#note that the test file is fnc dev partition
FILE=$DATA_DIR/test.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else

        # fnc-test lexicalized/plaintext
      wget https://osf.io/r5uvd/download -O $FILE

fi
fi
########fevercrossdomain delex where delexicalization was done using overlap aware (oa) technique (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
if [ "$TASK_TYPE" = "delex" ] && [ "$TASK_NAME" = "fevercrossdomain" ]  && [ "$SUB_TASK_TYPE" = "oa" ]; then
echo "found task type is lex and task name as fever cross domain and SUB_TASK_TYPE is oa"
echo $DATA_DIR
mkdir -p $DATA_DIR



FILE=$DATA_DIR/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/train.tsv -O $FILE
fi

FILE=$DATA_DIR/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/oa/dev.tsv -O $FILE
fi

#note that we are already replacing the file to be tested as test file. this way during run time you have to just
# load all files as is e.g #if do_predict is true load from folder fevercrossdomain/delex/test.tsv- which the code already does.

FILE=$DATA_DIR/test.tsv
if test -f "$FILE";then
echo "$FILE exists"
else
wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/oa/dev.tsv -O $FILE
fi
fi


########fevercrossdomain delex where delexicalixation was done with figer-specific technique (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
if [ "$TASK_TYPE" = "delex" ] && [ "$TASK_NAME" = "fevercrossdomain" ]  && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then

echo "found task type is delex and task name as fever cross domain and sub task type ==figerspecific"
echo $DATA_DIR
mkdir -p $DATA_DIR


FILE=$DATA_DIR/train.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer_specific/train.tsv -O $FILE
fi

FILE=$DATA_DIR/dev.tsv
if test -f "$FILE";then
    echo "$FILE exists"
else
   wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer_specific/dev.tsv -O $FILE
fi

#We are loading the cross domain dev file (eg: fnc-dev file) as the test partition
# This is just a hack so that the code produces/predicts results in single go, as opposed to having to reload a trained model
FILE=$DATA_DIR/test.tsv
if test -f "$FILE";then
echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/figer_specific/dev.tsv -O $FILE
fi
fi


####################################for cross domain student teacher, there will be two training files.-one for lex and another for delex


if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then
    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be figerspecific"

echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
#<<<<<<< #best_lex_model_helpful_vortex_with_mini_bert
    #wget https://osf.io/r6mdz/download    -O $FILE

    wget https://osf.io/r6mdz/download -O $FILE
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

    wget https://osf.io/azf6t/download -O $FILE #fever_dev_lex_4labels.txt
fi

fi

#############################


if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerabstract" ]; then
    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be figerabstract"

#note, train1.tsv will be the lexicalized version of the file(so the link below almost always never changes) while train2.tsv is the delexicalized, which can change
#based on the type of delexicalization algorithm used. eg:figerabstract
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
      wget https://osf.io/8shu4/download -O $FILE
fi

FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #uncomment this if you want to feed dev as lex. this is to check if lex model alone works fine from within student teacher
    wget https://osf.io/azf6t/download  -O $FILE
 
fi


#note that we are  replacing the test partition with cross domain dev partition(in this case it thus becomes the in-domain dev
# partition of fnc dataset).

FILE="$DATA_DIR/test.tsv"

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
    #wget https://osf.io/85h4z/download -O $FILE


      #uncomment this if you want to feed test as lex. this is to check if lex model alone works fine from within student teacher
       #note that this is actually the dev partition of fnc.
       wget https://osf.io/jfpbv/download -O $FILE
      #wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/figer_specific/dev.tsv   -O $FILE


fi

fi



####################################for 1 teacher 1 student ...cross domain , when all delex files are delexicalized with oa


#this is used for plain text runs. on fnccrossdomain. i.e trained on fnc (with 3 labels) and tested on fever-dev (as test partition) which has 3 labels


if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fnccrossdomain" ] ;then

    echo "found task type to be combined, taskname to be fnccrossdomain and subtasktype to be oa"

echo $DATA_DIR
mkdir -p $DATA_DIR

#this is fnc-train plain text
FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/dwef7/download -O $FILE
fi

#this is fnc-train delexicalized using figerspec. however for fnc cross domain lex run, we wont use this. refer flag_run_teacher_alone in trainer.py
FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/f2g4k/download -O $FILE
fi

#fnc-dev plain text version
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #wget https://osf.io/d9wnf/download  -O $FILE

    #will load fever-dev as dev for acl paper purposes
    wget https://osf.io/xdbh6/download -O $FILE
fi


FILE="$DATA_DIR/test.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
#will load fever-test plainteext for acl paper purposes
wget https://osf.io/q38pn/download -O $FILE

fi

#fi of if corresponding to 3t1s check
fi




####################################for testing cross domain aka fnccrossdomain- lexicalized data


if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fnccrossdomain" ] && [ "$SUB_TASK_TYPE1" = "lex" ] && [ "$SUB_TASK_TYPE2" = "lex" ]; then
    echo "found task type to be 3t1s, taskname to be fnccrossdomain and subtasktypes to be lex and lex"


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

FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training partition of fever delexicalized using figer abstract techique
   wget https://osf.io/mauqv/download -O $FILE
fi

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




######################multiple models of student teacher architecture in the other direction: train on fnc and test on fever

## model1:train1.tsv: fnc-train lexicalized data (3 labels)- [disagree, agree, nei]
# model 2: train2.tsv fnc-train delexicalized using figerspecific
# model 3: train3.tsv fnc-train delexicalized using oaner
#model 4 : train4.tsv fnc-train delexicalised using figerabstract
#------
# dev: ideally we should be running over 4 dev partitions similarly. too  much pain. also dev/indomain is more like a sanity check. if atleast one of them is in late 80s early 90s accuracy its good
#----
#test
## model1 will test on test1.tsv= fever-dev lexicalized data
# model 2: test2.tsv fever-dev delexicalized using figerspecific
# model 3: test3.tsv fever-dev delexicalized using oaner
#model 4 : test4.tsv fever-dev delexicalised using figerabstract


#########
#update: $SUB_TASK_TYPE means the type of delexicalization we will be using. however we are doing away with it, since
# in multiple teachers there are so many delexicalizations and hence no point trying to segregate a control flow based on that

#####
#update: $SUB_TASK_TYPE few_shot means, the training will be done with indomain training data mized with 10% data from target domain



if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fnccrossdomain" ] && [ "$SUBTASK_TYPE" = "few_shot" ] ; then
    echo "found task type to be 3t1s, taskname to be fnccrossdomain and subtask to be few shot"


echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/dwef7/download -O $FILE

    #in few shot we will try to paste a small amount of data (correspondingly delexicalized)from the target domain to the end of each training data.
    # here we download lex plain text version of fever in domain with 3 labels so as to paste at the end of fnc- 3 labels
    wget https://osf.io/k6tg7/download -O cross_domain_train1.tsv
    tail -2000 cross_domain_train1.tsv >> $FILE

fi



FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/f2g4k/download -O  $FILE
    # here we download figer specific delexicalized  version of fever in domain with 3 labels so as to paste at the end of fnc- 3 labels
    wget https://osf.io/utmn9/download -O cross_domain_train2.tsv
    tail -2000 cross_domain_train2.tsv >> $FILE
fi

# model 3: train3.tsv fnc-train delexicalized using oaner
FILE="$DATA_DIR/train3.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/3bjpz/download -O $FILE
   #oaner delexicalized fever indomain training data with 3 labels
   wget https://osf.io/aw8bf/download -O cross_domain_train3.tsv
    tail -2000 cross_domain_train3.tsv >> $FILE
fi


#model 4 : train4.tsv fnc-train delexicalised using figerabstract (aka fnc cross domain with 3 labels)
FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/kzdpb/download -O $FILE

   #fever indomain figer abstract delexicalized data 3labels
   wget https://osf.io/923sq/download -O cross_domain_train4.tsv
    tail -2000 cross_domain_train4.tsv >> $FILE
fi



#dev is dev partition of in-domain dataset, fnc, delexicalized with figerspecific
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

#note that we are loading the cross domain's dev partition as test partition here
# note: we are loading the dev partition of fnc dataset here..(which will be found in my osf.io account folder: student_teacher_fact_verification/all_input_files/fnc/in_domain/figerspecifid/dev.tsv)

    wget https://osf.io/msxfg/download -O $FILE


fi


####test partitions aka cross domain dev partitions
#plain text version of fever-dev
FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/xdbh6/download -O $FILE
fi


#figerspec delexicalized version of fever-dev
FILE="$DATA_DIR/test2.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/4n7b6/download -O $FILE

fi


#fever-dev delexicalized with oaner
FILE="$DATA_DIR/test3.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/5qupx/download -O $FILE
fi

#fever dev delexicalized with figer abstract
FILE="$DATA_DIR/test4.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/khc9e/download -O $FILE
fi
#fi of if corresponding to 3t1s check
fi


###########



if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUBTASK_TYPE" = "few_shot" ] ; then
    echo "found task type to be 3t1s, taskname to be fevercrossdomain and subtask to be few shot"
echo $DATA_DIR=
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
     wget https://osf.io/r6mdz/download -O $FILE

    # here we download lex plain text version of fnc in domain with 4 labels and append it at the end of fever4 labels in domain training data
    wget https://osf.io/a6tks/download -O cross_domain_train1.tsv
    tail -200 cross_domain_train1.tsv >> $FILE


fi



FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/8shu4/download -O  $FILE


    # here we download delex figerspecific version of fnc in domain with 4 labels and append it at the end of fever4 labels in domain training data
    wget https://osf.io/78vbs/download -O cross_domain_train2.tsv
    tail -200 cross_domain_train2.tsv >> $FILE


fi

# model 3: train3.tsv fnc-train delexicalized using oaner
FILE="$DATA_DIR/train3.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/uwcxs/download -O $FILE

    # here we download delex oaner version of fnc in domain with 4 labels and append it at the end of fever4 labels in domain training data
    wget https://osf.io/djkg3/download -O cross_domain_train3.tsv
    tail -200 cross_domain_train3.tsv >> $FILE


fi


#model 4 : train4.tsv fnc-train delexicalised using figerabstract (aka fnc cross domain with 3 labels)
FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/mauqv/download -O $FILE
    # here we download figerabstract delexicalised/version of fnc in domain with 4 labels and append it at the end of fever4 labels in domain training data
    wget https://osf.io/2grqh/download -O cross_domain_train4.tsv
    tail -200 cross_domain_train4.tsv >> $FILE

fi



#dev is dev partition of in-domain dataset, fnc, delexicalized with figerspecific
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #loading fever-dev-figerspecific as dev dataset. note: ideally we must have four dev files also. but i am ignoring that since main goal is to test on cross domain, which we are doing in test partition
    wget https://osf.io/r5pz3/download -O $FILE
fi


####test partitions aka cross domain dev partitions
FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jfpbv/download -O $FILE
fi


FILE="$DATA_DIR/test2.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jx32m/download -O $FILE

fi


FILE="$DATA_DIR/test3.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/b4qau/download -O $FILE
fi

FILE="$DATA_DIR/test4.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/m4dzs/download -O $FILE
fi
#fi of if corresponding to 3t1s check
fi


#############



if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fnccrossdomain" ] ; then
    echo "found task type to be 3t1s, taskname to be fnccrossdomain and subtasktypes to be figerspecific and oa"
echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
     wget https://osf.io/dwef7/download -O $FILE
fi



FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/f2g4k/download -O  $FILE
fi

# model 3: train3.tsv fnc-train delexicalized using oaner
FILE="$DATA_DIR/train3.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/3bjpz/download -O $FILE
fi


#model 4 : train4.tsv fnc-train delexicalised using figerabstract (aka fnc cross domain with 3 labels)
FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/kzdpb/download -O $FILE
fi



#dev is dev partition of in-domain dataset, fnc, delexicalized with figerspecific
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

#note that we are loading the cross domain's dev partition as test partition here
# note: we are loading the dev partition of fnc dataset here..(which will be found in my osf.io account folder: student_teacher_fact_verification/all_input_files/fnc/in_domain/figerspecifid/dev.tsv)

    wget https://osf.io/msxfg/download -O $FILE
fi


####test partitions aka cross domain dev partitions
#plain text version of fever-dev
FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/xdbh6/download -O $FILE
fi


#figerspec delexicalized version of fever-dev
FILE="$DATA_DIR/test2.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/4n7b6/download -O $FILE

fi


#fever-dev delexicalized with oaner
FILE="$DATA_DIR/test3.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/5qupx/download -O $FILE
fi

#fever dev delexicalized with figer abstract
FILE="$DATA_DIR/test4.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/khc9e/download -O $FILE
fi
#fi of if corresponding to 3t1s check
fi


##############


if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fevercrossdomain" ] ; then
    echo "found task type to be 3t1s, taskname to be fevercrossdomain and subtasktypes to be figerspecific and oa"
echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
     wget https://osf.io/r6mdz/download -O $FILE
fi



FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/8shu4/download -O  $FILE
fi

# model 3: train3.tsv fnc-train delexicalized using oaner
FILE="$DATA_DIR/train3.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/uwcxs/download -O $FILE
fi


#model 4 : train4.tsv fnc-train delexicalised using figerabstract (aka fnc cross domain with 3 labels)
FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

   wget https://osf.io/mauqv/download -O $FILE
fi



#dev is dev partition of in-domain dataset, fnc, delexicalized with figerspecific
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #loading fever-dev-figerspecific as dev dataset. note: ideally we must have four dev files also. but i am ignoring that since main goal is to test on cross domain, which we are doing in test partition
    wget https://osf.io/r5pz3/download -O $FILE
fi


####test partitions aka cross domain dev partitions
FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jfpbv/download -O $FILE
fi


FILE="$DATA_DIR/test2.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jx32m/download -O $FILE

fi


FILE="$DATA_DIR/test3.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/b4qau/download -O $FILE
fi

FILE="$DATA_DIR/test4.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
      wget https://osf.io/m4dzs/download -O $FILE
fi

fi

##############