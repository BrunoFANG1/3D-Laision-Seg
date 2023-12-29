#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --export=ALL

python ./test.py
