#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="test_new_jets"
#SBATCH --time=10:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6

srun uv run test_new_jets.py 
