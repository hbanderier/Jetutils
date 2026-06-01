#!/bin/bash
#SBATCH --job-name="Download temp"
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=150G
#SBATCH --cpus-per-task=2

# Your code below this line
mamba activate env12
srun python add_temp_to_flat_wind.py
