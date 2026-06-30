#!/bin/bash
#------------------------
#SBATCH --account="gratis"
#------------------------
#SBATCH --qos="job_gratis"
#SBATCH --job-name="download_uvnew"
#SBATCH --time=24:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

srun uv run download_ERA5.py 
