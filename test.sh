#!/bin/bash
#SBATCH -A rsteven1_gpu
#SBATCH --job-name=3D_Unet_NewData
#SBATCH –output="test.log"
#SBATCH –error="err.log"
#SBATCH --nodes=1
#SBATCH --partition=ica100
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xfang22@jh.edu
#SBATCH --mem-per-cpu=8G 

python 