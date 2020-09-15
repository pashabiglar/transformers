[BERT]
model_name_or_path="bert-base-cased"
task_name="fevercrossdomain"
do_train=True
do_eval=True
do_predict=True
data_dir="/Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/delex/figerspecific/toydata/"
max_seq_length="128"
per_device_eval_batch_size="16"
per_device_train_batch_size="16"
learning_rate="1e-5"
num_train_epochs="2"
output_dir="/Users/mordor/research/huggingface/mithun_scripts/output/fever/fevercrossdomain/delex/figerspecific/bert-base-cased/128/"
overwrite_output_dir=True
weight_decay="0.01"
adam_epsilon="1e-6"
evaluate_during_training=True
task_type="delex"
subtask_type="figerspecific"
machine_to_run_on="laptop"
toy_data_dir_path="/Users/mordor/research/huggingface/src/transformers/data/datasets/fever/fevercrossdomain/delex/figerspecific/toydata/"
#what are the scores that you should assert against. i.e the ones we got when we ran the toy data alone
fever_in_domain_accuracy_on_toy_data_17_datapoints=0.625
fever_cross_domain_accuracy_on_toy_data_17_datapoints=0.75
fever_cross_domain_fncscore_on_toy_data_17_datapoints=0.80