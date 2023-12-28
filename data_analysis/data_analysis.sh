#!/bin/bash
#SBATCH -A rsteven1_gpu
#SBATCH --job-name=data_analysis
#SBATCH --error=./logs/err.out
#SBATCH --output=./logs/log.out
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --partition=ica100
#SBATCH --time=04:10:00
#SBATCH --mem-per-cpu=28G 

python data_analysis.py

