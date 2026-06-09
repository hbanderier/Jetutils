#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="fix_cat"
#SBATCH --time=05:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=22G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

srun uv run fix_cat.py 
