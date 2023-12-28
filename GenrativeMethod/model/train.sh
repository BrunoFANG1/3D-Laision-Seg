#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --export=ALL

python /home/bruno/3D-Laision-Seg/GenrativeMethod/model/CT2MRI_3DGAN/main_multiGPU.py
