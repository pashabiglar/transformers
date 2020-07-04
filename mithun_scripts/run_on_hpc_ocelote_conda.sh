#!/usr/bin/env bash

####PBS -q windfall
#####PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1

# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
### Specify a name for the job
#PBS -N bert_uncased_figer_standalone
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:shared
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=427:00:00
### Walltime is how long your job will run
#PBS -l walltime=15:15:00
### Joins standard error and standard out
#PBS -j oe


cd /home/u11/mithunpaul/

module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5
module load anaconda/2020/2020.02

conda create --name huggingface2 python=3.8

#this is the only line you need if you already have a virtual_env set up
source activate huggingface2



#pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92-f https://download.pytorch.org/whl/torch_stable.html



#####my code part

cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
export PYTHONPATH="/home/u11/mithunpaul/huggingfacev2/src"
pip install --upgrade pip
pip install wandb
pip install -r requirements.txt

WANDB_API_KEY=de268c256c2d4acd9085ee4e05d91706c49090d7
#wandb login
#export WANDB_NAME="from_hpc"
./run_all.sh







