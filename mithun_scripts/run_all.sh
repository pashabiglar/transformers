#!/bin/bash



export MACHINE_TO_RUN_ON="laptop" #options include [laptop, hpc,clara]
export EPOCHS=1
if [ $# -gt 1 ]
then
   echo "number of args is $#"
   if [ $1 == "--epochs_to_run" ]; then
         export EPOCHS="$2"
    fi
else
  echo "number of args is not greater than 1"
  export EPOCHS=1
fi





if [ $# -gt 2 ]; then
if [ $3 == "--machine_to_run_on" ]; then
        export MACHINE_TO_RUN_ON=$4
fi
fi



if [ $# -gt 3 ]; then
if [ $5 == "--use_toy_data" ]; then
        export USE_TOY_DATA=$6
fi
fi


if [ $# -gt 7 ]; then
if [ $7 == "--download_fresh_data" ]; then
        export DOWNLOAD_FRESH_DATA=$8
fi
fi


if [ $MACHINE_TO_RUN_ON == "hpc" ]; then
        wandb on
        wandb online
        export OUTPUT_DIR_BASE="/home/u11/mithunpaul/xdisk/fnc2fever_gl_bert_base_cased_rs3082_wt11/output"
        export DATA_DIR_BASE="/home/u11/mithunpaul/xdisk/fnc2fever_gl_bert_base_cased_rs3082_wt11/data"
fi

if [ $MACHINE_TO_RUN_ON == "laptop" ]; then
        wandb off
        export DATA_DIR_BASE="/Users/mordor/research/huggingface/src/transformers/data/datasets"
        export OUTPUT_DIR_BASE="/Users/mordor/research/huggingface/mithun_scripts/output"
        export PYTHONPATH="/Users/mordor/research/huggingface/src/"
fi

if [ $MACHINE_TO_RUN_ON == "clara" ]; then
        wandb on
        wandb online
        export OUTPUT_DIR_BASE="/work/mithunpaul/huggingface_bertmini_multiple_teachers_v1/output"
        export DATA_DIR_BASE="/work/mithunpaul/huggingface_bertmini_multiple_teachers_v1/data"

fi


echo "MACHINE_TO_RUN_ON=$MACHINE_TO_RUN_ON"
echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
echo "DATA_DIR_BASE=$DATA_DIR_BASE"
echo "EPOCHS=$EPOCHS"


export DATASET="fever" #the name of the home/in-domain dataset . options include [fever, fnc]

# Will your model be a stand alone model (lex,delex) or a student teacher architecture one (combined) with two models,
#update: if using group_learning setup (more than 2 models), use :3t1s
export TASK_TYPE="3t1s" #[lex, delex, combined, 3t1s]

#if your TASK_TYPE is combined,  what types of delexiccalizations will your student teacher model be using.
# also if you want to add fewshot learning to your models (irrespective of the number of models), use: few_shot
#note: in case of having more than 2 models, (i.e group learning), this variable is not checked (i.e the order of delexicalizations are fixed)
# unless you are using few_shot+3t1s
#export SUBTASK_TYPE="few_shot" #['few_shot',"oa","figer_specific", "figer_abstract"]

export TASK_NAME="fnccrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export BERT_MODEL_NAME="bert-base-cased" #options include things like [bert-base-uncased,bert-base-cased , minibert(google/bert_uncased_L-12_H-128_A-2)] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"

export basedir="$DATA_DIR_BASE/$DATASET"
export PYTHONPATH="../src"
export DATA_DIR="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE"
export TOY_DATA_DIR="toydata"
export TOY_DATA_DIR_PATH="$DATA_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$TOY_DATA_DIR/"
export OUTPUT_DIR="$OUTPUT_DIR_BASE/$DATASET/$TASK_NAME/$TASK_TYPE/$BERT_MODEL_NAME/$MAX_SEQ_LENGTH/"
echo $OUTPUT_DIR




echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "value of epochs is $EPOCHS"


echo "$DOWNLOAD_FRESH_DATA"

if [ $DOWNLOAD_FRESH_DATA == "true" ]; then
    echo "found DOWNLOAD_FRESH_DATA is true "
    rm -rf $DATA_DIR
    ./get_fever_fnc_data.sh
    ./convert_to_mnli_format.sh
fi
echo "value of toy_data_path is $TOY_DATA_DIR_PATH"
#create a small part of data as toy data. this will be used to run regresssion tests before the actual run starts
./reduce_size.sh  --data_path $TOY_DATA_DIR_PATH





echo "value of DATA_DIR is $DATA_DIR"


echo "done with data download  TOY_DATA_DIR_PATH now is $TOY_DATA_DIR_PATH"


#use a smaller toy data to test


if  [ "$USE_TOY_DATA" = true ]; then
        DATA_DIR=$TOY_DATA_DIR_PATH
        echo "found USE_TOY_DATA is true"
fi

echo "done with data download part . datapath now is $DATA_DIR"



export CUDA_VISIBLE_DEVICES=2
set CUDA_VISIBLE_DEVICES=2


export args="--model_name_or_path $BERT_MODEL_NAME   --task_name $TASK_NAME      --do_train   --do_eval   --do_predict    \
--data_dir $DATA_DIR    --max_seq_length $MAX_SEQ_LENGTH      --per_device_eval_batch_size=16        --per_device_train_batch_size=16       \
--learning_rate 1e-5      --num_train_epochs $EPOCHS     --output_dir $OUTPUT_DIR --overwrite_output_dir  \
--weight_decay 0.01 --adam_epsilon 1e-6  --evaluate_during_training \
--task_type $TASK_TYPE --machine_to_run_on $MACHINE_TO_RUN_ON --toy_data_dir_path $TOY_DATA_DIR_PATH  \
--overwrite_cache --do_train_student_teacher --total_no_of_models_including_student_and_its_teachers 4 --total_no_of_test_datasets 4 --classification_loss_weight 11"



##test cases
# ./run_training_tests.sh
#./run_loading_tests.sh



#actual code runs
./run_glue.sh
#./load_model_test.sh



