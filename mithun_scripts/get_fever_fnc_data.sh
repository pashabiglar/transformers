#!/bin/bash







#######fevercrossdomain lex (training and dev will be in fever (with 4 labels), and test on fnc-dev partition)
if [ "$TASK_TYPE" = "lex" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "figerspecific" ]; then

echo "found task type is lex and task name as fever cross domain"


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

    wget https://osf.io/r6mdz/download -O $FILE
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

    #uncomment this if you want to feed lexicalized version of the dataset (fever-dev) as  dev partition. this is useful when you want to sanity check  how a lexicalized model is performing
    #wget https://osf.io/azf6t/download -O $FILE

    wget https://osf.io/r5pz3/download -O $FILE

fi


#note that we are  replacing the test partition with cross domain dev partition(in this case. it thus becomes the in-domain dev partition of fnc dataset).

FILE="$DATA_DIR/test.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else

      # if you want to use the lexicalized version of the dataset (fnc-dev) as the test partition.
      # this is useful when you want to sanity check  how a lexicalized model is performing

      #fnc-dev lexicalized plaintext
      #wget https://osf.io/qs4u6/download -O $FILE

      # fnc-dev delexicalized using figerspecific
      #wget https://osf.io/jx32m/download   -O $FILE

      # fnc-test delexicalized using figerspecific
      wget https://osf.io/jentp/download   -O $FILE


        # fnc-test lexicalized/plaintext
      #wget https://osf.io/r5uvd/download -O $FILE




fi

fi


####################################for cross domain student teacher, there will be two training files.-one for mod1 and another for mod2


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
      wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer-abstract/train.tsv -O $FILE
fi

FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else

    wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FEVER/cross-domain/figer-abstract/dev.tsv -O $FILE

fi


#note that we are  replacing the test partition with cross domain dev partition(in this case it thus becomes the in-domain dev
# partition of fnc dataset).

FILE="$DATA_DIR/test.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else

      #uncomment this if you want to feed dev and test as lex. this is used when you want to check if lex model alone works fine from within student teacher
      #wget https://storage.googleapis.com/fact_ve®©rification_mithun_files/TSV/FNC/in-domain/lex/dev.tsv -O $FILE
      wget https://storage.googleapis.com/fact_verification_mithun_files/TSV/FNC/in-domain/figer_specific/dev.tsv   -O $FILE



fi

fi



####################################for cross domain student teacher, when delex files are delexicalized with oa





if [ "$TASK_TYPE" = "combined" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE" = "oa" ]; then
    echo "found task type to be combined, taskname to be feverCrossDomain and subtasktype to be oa"

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



####################################for 2 teachers one student, where
# teacher1 : sees data in lexicalized form/(train1.tsv)
# student : sees data delexicalized with figerspecific (train2.tsv)
# teacher2 : sees data delexicalized with oa (overlap aware) ner technique (train3.tsv)
# teacher : sees data delexicalized with figerabstract technique (train4.tsv)




if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fevercrossdomain" ] && [ "$SUB_TASK_TYPE1" = "figerspecific" ] && [ "$SUB_TASK_TYPE2" = "oa" ]; then
    echo "found task type to be 2t1s, taskname to be feverCrossDomain and subtasktypes to be figerspecific and oa"

echo $DATA_DIR
mkdir -p $DATA_DIR

FILE="$DATA_DIR/train1.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training with lexicalized/plain text version of fever
     wget https://osf.io/r6mdz/download  -O $FILE
fi


FILE="$DATA_DIR/train2.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training partition of fever delexicalized using figer specific techique
    wget https://osf.io/8shu4/download -O $FILE
fi


FILE="$DATA_DIR/train3.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training partition of fever delexicalized using oaner techique
   wget https://osf.io/uwcxs/download -O $FILE
fi


FILE="$DATA_DIR/train4.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    #this is the training partition of fever delexicalized using oaner techique
   wget https://osf.io/mauqv/download -O $FILE
fi



#dev is dev partition of in-domain dataset, fever, delexicalized with figerspecific
FILE="$DATA_DIR/dev.tsv"
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/r5pz3/download -O $FILE
fi

#note that we are loading the cross domain's dev partition as test partition here
# note: we are loading the dev partition of fnc dataset here..(which will be found in my osf.io account folder: student_teacher_fact_verification/all_input_files/fnc/in_domain/figerspecifid/dev.tsv)

#update : trying to use each of the n models in n corresponding delexicalized veersions of the dataset.
# each tst partition number will match to corresponding deeleexicalization used in training. so test1.tsv will be dev partition of cross domain plain text, test2.tsv will be figerspec version etc


#plain text version of fnc-dev
FILE="$DATA_DIR/test1.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jfpbv/download -O $FILE
fi


#figerspec delexicalized version of fnc-dev
FILE="$DATA_DIR/test2.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/jx32m/download -O $FILE
fi


#fnc-dev delexicalized with oaner
FILE="$DATA_DIR/test3.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/b4qau/download -O $FILE
fi


#fnc-dev delexicalized with figerabstract
FILE="$DATA_DIR/test4.tsv"
if test -f "$FILE";then
echo "$FILE exists"
else
     wget https://osf.io/m4dzs/download -O $FILE
fi





#fi of if corresponding to 3t1s check
fi




######################multiple models of student teacher architecture in the other direction: train on fnc and test on fever
#note: subtasktype1, subtasktype2 etc is vestigial from old code. now we are using 4 models each trained on as follows
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



if [ "$TASK_TYPE" = "3t1s" ] && [ "$TASK_NAME" = "fnccrossdomain" ] && [ "$SUB_TASK_TYPE1" = "figerspecific" ] && [ "$SUB_TASK_TYPE2" = "oa" ]; then
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

    #wget https://osf.io/msxfg/download -O $FILE
        #loading fever-dev-oaner as dev dataset . for acl final paper.
    wget https://osf.io/5qupx/download -O $FILE
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
      #wget https://osf.io/5qupx/download -O $FILE
      #loading fever-testoaner-3labels as test3 for acl paper
      wget https://osf.io/u2kw8/download -O $FILE
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