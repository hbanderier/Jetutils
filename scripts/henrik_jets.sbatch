#!/bin/bash
#SBATCH --job-name="henrik_jets"
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8

srun uv run henrik_jets.py 
