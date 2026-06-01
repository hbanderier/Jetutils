#!/bin/bash
#SBATCH --job-name="pvs_to_da"
#SBATCH --time=05:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate env12
srun python pvs_to_da.py 
