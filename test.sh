#!/bin/bash
#SBATCH -A rsteven1_gpu
#SBATCH --job-name=3D_Unet_NewData
#SBATCH --error=./logs/err.out
#SBATCH --output=./logs/log.out
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --partition=a100
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xfang22@jh.edu
#SBATCH --mem-per-cpu=8G 

python test_run.py
