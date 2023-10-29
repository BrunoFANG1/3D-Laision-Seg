#!/bin/bash
#SBATCH -A rsteven1_gpu
#SBATCH --job-name=3D_Unet_NewData
#SBATCH --error=./logs/err.out
#SBATCH --output=./logs/log.out
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=03:10:00
#SBATCH --mem-per-cpu=28G 

python test_run.py

