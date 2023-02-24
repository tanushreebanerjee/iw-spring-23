#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=load_datasets_iw_vqa_test_1000
#SBATCH --ntasks-per-node=1
#SBATCH --output=load_datasets.txt
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH -t 96:00:00
#SBATCH --mem 64G
# sends mail when process begins, and
# when it ends. Make sure you define your email
# address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tb21@princeton.edu


source ~/.bashrc

conda activate iw_vqa

cd /n/fs/nlp-runzhey/diplomacy/iw-spr-23/iw-spring-23/utils

python load_dataset.py