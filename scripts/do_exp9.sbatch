#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="exp9"
#SBATCH --time=04:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

srun uv run do_exp9.py 
