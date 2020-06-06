#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=8gb:ngpus=1
### Specify a name for the job
#PBS -N 1_contextual
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=1344:00:00
### Walltime is how long your job will run
#PBS -l walltime=48:00:00
#PBS -e /home/u11/mithunpaul/xdisk/huggingface/hpc_errors_outputs/
#PBS -o /home/u11/mithunpaul/xdisk/huggingface/hpc_errors_outputs/

#####module load cuda80/neuralnet/6/6.0
#####module load cuda80/toolkit/8.0.61
module load singularity/2/2.6.1

cd $PBS_O_WORKDIR

#train
export PYTHONPATH="/home/u11/mithunpaul/xdisk/huggingface/transformers/src/"
export GLUE_DIR="/home/u11/mithunpaul/xdisk/huggingface/transformers/src/transformers/data/datasets/fever/feverindomain/lex/"
export TASK_NAME=feverindomain

singularity exec --nv  /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img bash /xdisk/msurdeanu/mithunpaul/huggingface/transformers/mithun_scripts/install_stuff.sh
singularity exec --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img python /xdisk/msurdeanu/mithunpaul/huggingface/transformers/examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased     --task_name $TASK_NAME      --do_train     --do_eval      --data_dir $GLUE_DIR    --max_seq_length 128      --per_device_eval_batch_size=8        --per_device_train_batch_size=8        --learning_rate 2e-5      --num_train_epochs 5.0      --output_dir /tmp/$TASK_NAME --overwrite_output_dir






