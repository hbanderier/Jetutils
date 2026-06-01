#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="cesm_jets"
#SBATCH --time=10:00:00
#SBATCH --mem=140G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6

srun uv run cesm_jets.py 
