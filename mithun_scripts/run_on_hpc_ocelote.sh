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
#singularity run --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/BERT_QA/bert_latest/bert/run_classifier_ARC_DETAILED_sandeep.py --task_name=fevercd --do_predict=true --data_dir=$GLUE_DIR --vocab_file=$MODEL_DIR/vocab.txt --bert_config_file=$MODEL_DIR/bert_config.json --init_checkpoint=$TRAINED_MODEL --max_seq_length=64 --output_dir=$OUT_DIR_2 --do_lower_case=true
singularity run --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img /home/u11/mithunpaul/xdisk/huggingface/transformers/mithun_scripts/get_fever_fnc_data.sh
singularity run --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img /home/u11/mithunpaul/xdisk/huggingface/transformers/mithun_scripts/reduce_size.sh
singularity run --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img /home/u11/mithunpaul/xdisk/huggingface/transformers/mithun_scripts/convert_to_mnli_format.sh
singularity run --nv /xdisk/msurdeanu/mithunpaul/BERT_REPLICATION/ocelote_BERT_singularity.img /home/u11/mithunpaul/xdisk/huggingface/transformers/mithun_scripts/run_glue.sh

