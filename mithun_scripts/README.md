### how to run 

`mkdir factverfication`

`cd factverification`

`mkdir code`

`mkdir output`

`mkdir data`

`cd code`

`git clone git@github.com:mithunpaul08/transformers.git .`

`cd mithun_scripts`


change the paths in the following code to the corresponding ones.
also pick a name you want to run on as $MACHINE_TO_RUN_ON

```
if [ $MACHINE_TO_RUN_ON == "clara" ]; then
        wandb on
        wandb online
        export OUTPUT_DIR_BASE="/work/mithunpaul/huggingface_bertmini_multiple_teachers_v1/output"
        export DATA_DIR_BASE="/work/mithunpaul/huggingface_bertmini_multiple_teachers_v1/data"
fi

```

also change the paramaters of below values

```
export DATASET="fever" #options include [fever, fnc]
export TASK_TYPE="3t1s" #options for task type include lex,delex,and combined"". combined is used in case of student teacher architecture which will load a paralleldataset from both mod1 and mod2 folders
export SUBTASK_TYPE="few_shot"
export TASK_NAME="fevercrossdomain" #options for TASK_NAME  include fevercrossdomain,feverindomain,fnccrossdomain,fncindomain
export BERT_MODEL_NAME="google/bert_uncased_L-12_H-128_A-2" #options include things like [bert-base-uncased,bert-base-cased] etc. refer src/transformers/tokenization_bert.py for more.
export MAX_SEQ_LENGTH="128"
```

then run the code using:

`bash run_all.sh --epochs_to_run 25 --machine_to_run_on hpc --use_toy_data false --download_fresh_data true `

note: change the $MACHINE_TO_RUN_ON to whatever you picked aboce

## to run 1+ models
pass command line args:
total_no_of_models_including_student_and_its_teachers=1
total_no_of_test_datasets=1


to run 1teacher 1 student pass same as 2.

Note, here is the mapping 
1= just run lex alone
2= lex as teacher, student will have data that is delexicalized with figerspecific
3= 1t, 1student as before+ another student with delexicalized using oaner
4= 4 models, with : lex, figerspecific, oaner, figerabstract


#### for internal uofa reference

scripts to run on each of the machines:

`bash run_all.sh --epochs_to_run 25 --machine_to_run_on hpc --use_toy_data false --download_fresh_data true #options include [laptop, hpc,clara]`

`bash run_all.sh --epochs_to_run 2 --machine_to_run_on laptop --use_toy_data true --download_fresh_data true`

`bash run_all.sh --epochs_to_run 55 --machine_to_run_on clara --use_toy_data false --download_fresh_data true`

`bash run_all.sh --epochs_to_run 25 --machine_to_run_on clara --use_toy_data false --download_fresh_data true`