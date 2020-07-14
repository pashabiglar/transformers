#!/usr/bin/env bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
### Specify a name for the job
#PBS -N hf_teacher_student
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=359:20:00
### Walltime is how long your job will run
#PBS -l walltime=48:00:00
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

export PYTHONPATH="/home/u11/mithunpaul/huggingfacev2/src"
pip install --upgrade pip
#pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92-f https://download.pytorch.org/whl/torch_stable.html



#####my code part

#pip install -r /xdisk/msurdeanu/mithunpaul/huggingface/transformers/examples/requirements.txt
#
#cd /xdisk/msurdeanu/mithunpaul/huggingface/transformers/mithun_scripts/
#bash /xdisk/msurdeanu/mithunpaul/huggingface/transformers/mithun_scripts/run_all.sh



#one which worked at noon june 6th 2020
cd /home/u11/mithunpaul/huggingfacev2/mithun_scripts/
pip install -r requirements.txt
#export PYTHONPATH="/xdisk/msurdeanu/mithunpaul/huggingface//transformers/src/"


#export GLUE_DIR="/xdisk/msurdeanu/mithunpaul/huggingface/transformers/src/transformers/data/datasets/fever/feverindomain/lex/"
#export TASK_NAME=fevercrossdomain

bash run_all.sh







