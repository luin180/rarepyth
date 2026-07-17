#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p qdagnormal
#SBATCH -J name
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

module load nvidia/cuda/12.2
module load apps/apptainer/1.3.4

apptainer exec $HOME/pytorch_env.sif python mace_train.py
