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

note: change the $MACHINE_TO_RUN_ON to whatever you picked above

### to run just 1 model
remove command line argument `do_train_student_teacher`
pass command line args:

total_no_of_models_including_student_and_its_teachers=1

total_no_of_test_datasets=1
task_type=[lex or delex]

e.g.,`--total_no_of_models_including_student_and_its_teachers 1 --total_no_of_test_datasets 1 --task_type lex`


## to run 1+ models
pass command line args:
total_no_of_models_including_student_and_its_teachers=1
total_no_of_test_datasets=1


e.g.,
`--do_train_student_teacher  --total_no_of_models_including_student_and_its_teachers 1 --total_no_of_test_datasets 1`



Note, here is the mapping 
1= just run lex alone
2= lex as teacher, student will have data that is delexicalized with figerspecific
3= 1t, 1student as before+ another student with delexicalized using oaner
4= 4 models, with : lex, figerspecific, oaner, figerabstract

## other command line arguments

`--overwrite_cache` : add this if you want the cache to be overwritten. Usually the 
tokenized data is stored and reused from the cache. Especially when `--download_fresh_data` is set
to True, it is imperative that `--overwrite_cache` is added to the list of arguments.



## Typical examples of command line arguments. 

- to run a stand alone model which trains on lexicalized plain text data. This is the classic training of any neural network. 

```
--model_name_or_path google/bert_uncased_L-12_H-128_A-2 --task_name fevercrossdomain --do_train --do_eval --do_predict --data_dir /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/lex/toydata/ --max_seq_length 128 --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 2 --output_dir /Users/mordor/research/huggingface/mithun_scripts/output/fever/fevercrossdomain/lex/google/bert_uncased_L-12_H-128_A-2/128/ --overwrite_output_dir --weight_decay 0.01 --adam_epsilon 1e-6 --evaluate_during_training --task_type lex --machine_to_run_on laptop --toy_data_dir_path /Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/lex/toydata/ --overwrite_cache --total_no_of_models_including_student_and_its_teachers 1 --total_no_of_test_datasets 1

```

#### for internal uofa reference

scripts to run on each of the machines:

`bash run_all.sh --epochs_to_run 25 --machine_to_run_on hpc --use_toy_data false --download_fresh_data true #options include [laptop, hpc,clara]`

`bash run_all.sh --epochs_to_run 2 --machine_to_run_on laptop --use_toy_data true --download_fresh_data true`

`bash run_all.sh --epochs_to_run 55 --machine_to_run_on clara --use_toy_data false --download_fresh_data true`

`bash run_all.sh --epochs_to_run 25 --machine_to_run_on clara --use_toy_data false --download_fresh_data true`