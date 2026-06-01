#!/bin/bash
#SBATCH --job-name="zarrify"
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=2

srun uv run zarrify.py 
