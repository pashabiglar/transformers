#!/bin/bash
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
#PBS -W group_list=msurdeanu
#PBS -l walltime=48:00:00
#PBS -e /home/u11/mithunpaul/xdisk/huggingface/hpc_errors_outputs/
#PBS -o /home/u11/mithunpaul/xdisk/huggingface/hpc_errors_outputs/


cd /home/u11/mithunpaul/
module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
rm -rf my_virtual_env
mkdir my_virtual_env
python3 -m venv my_virtual_env

#this is the only line you need if you already have a virtual_env set up
source my_virtual_env/bin/activate


pip install --upgrade pip
#pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt



#####my code part
cd /xdisk/msurdeanu/mithunpaul/huggingface/transformers/mithun_scripts/
export PYTHONPATH="/xdisk/msurdeanu/mithunpaul/huggingface//transformers/src/"


export GLUE_DIR="/xdisk/msurdeanu/mithunpaul/huggingface/transformers/src/transformers/data/datasets/fever/feverindomain/lex/"
export TASK_NAME=feverindomain

bash /xdisk/msurdeanu/mithunpaul/huggingface/transformers/mithun_scripts/run_all.sh






