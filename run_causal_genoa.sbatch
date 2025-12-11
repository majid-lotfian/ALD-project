#!/bin/bash
#SBATCH --job-name=causal_gen
#SBATCH --output=logs/causal_gen_%j.out
#SBATCH --error=logs/causal_gen_%j.err
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1

# ------------------------------------------------------------
# Load required modules
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Activate your venv
source $HOME/.venvs/ald-py312/bin/activate

# Go to the project directory
cd "/gpfs/work1/0/prjs1732/LabGPT/ALD"

# Create log directory if missing
mkdir -p logs

# Run the Python script
python 2.4.2.CausalModelSyntheticDataV2.py
