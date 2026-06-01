#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate env12
srun python download_ERA5.py
