#!/usr/bin/env bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
### Specify a name for the job
#PBS -N bug_fix_parallelization_issues
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:shared
### Walltime is how long your job will run
#PBS -l walltime=1:00:00
### Joins standard error and standard out
#PBS -j oe




cd /home/u11/mithunpaul/

module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
#rm -rf my_virtual_env
#mkdir my_virtual_env
python3 -m venv my_virtual_env

#this is the only line you need if you already have a virtual_env set up
source my_virtual_env/bin/activate
export PYTHONPATH="/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/src"

#for clara
#export PYTHONPATH="/work/mithunpaul/huggingface_fix_paralellism_per_epoch_issue/src"
pip install --upgrade pip



cd /home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/examples


pip install -r examples/requirements.txt

cd /home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/mithun_scripts/

bash run_all.sh --epochs_to_run 1 --machine_to_run_on hpc #options include [laptop, hpc,clara]
# for server clara
#bash run_all.sh --epochs_to_run 1 --machine_to_run_on clara






