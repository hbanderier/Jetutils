#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="cc"
#SBATCH --time=08:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=37G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

srun uv run do_cc.py 
