#!/bin/bash
#SBATCH --account=p1605-25-3  # project code
#SBATCH -J "Evaluation"  # name of job in SLURM
#SBATCH --partition=gpu  # selected partition
#SBATCH --gres=gpu:1  # total gpus
#SBATCH --nodes=1  # number of used nodes
#SBATCH --time=5:00:00  # time limit for a job
#SBATCH -o stdout.%J.out  # standard output
#SBATCH -e stderr.%J.out  # error output

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate MultiCW-finetune

python $HOME/MultiCW/evaluation.py

