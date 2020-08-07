#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --oversubscribe
#SBATCH --time=00:10:00
#SBATCH --job-name=slurm-standard-test
#SBATCH --account=msurdeanu
#SBATCH --partition=standard
#SBATCH --output=slurm-standard-test.out





cd /home/u11/mithunpaul/

module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
#rm -rf my_virtual_env
#mkdir my_virtual_env
python3 -m venv my_virtual_env

#this is the only line you need if you already have a virtual_env set up
source my_virtual_env/bin/activate
export PYTHONPATH="/home/u11/mithunpaul/xdisk/huggingface_bert_run_expts_on_puma/src"

#for clara
#export PYTHONPATH="/work/mithunpaul/huggingface_fix_paralellism_per_epoch_issue/src"
pip install --upgrade pip



cd /home/u11/mithunpaul/xdisk/huggingface_bert_run_expts_on_puma/examples


pip install -r requirements.txt

cd /home/u11/mithunpaul/xdisk/huggingface_bert_run_expts_on_puma/mithun_scripts/

bash run_all.sh --epochs_to_run 1 --machine_to_run_on hpc #options include [laptop, hpc,clara]
# for server clara
#bash run_all.sh --epochs_to_run 2 --machine_to_run_on clara






