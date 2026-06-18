#!/bin/bash
# Two-node version: 2 nodes × 4 GPUs = 8 GPUs total.
# Effective batch = 256 × 8 = 2048 per step.
#SBATCH --job-name=ald_pretrain_v3_2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=460G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --output=logs/pretrain_2n_%j.out
#SBATCH --error=logs/pretrain_2n_%j.err

mkdir -p logs

module purge
module load 2024

eval "$(conda shell.bash hook)"
conda activate ald

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "MASTER_ADDR=$MASTER_ADDR  WORLD_SIZE=$WORLD_SIZE"
echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"

python -c "import torch; assert hasattr(torch.amp, 'GradScaler'), 'check failed'" || {
    echo "ERROR: torch.amp.GradScaler not found. Ensure the 'ald' conda env activated correctly."
    exit 1
}

cd $SLURM_SUBMIT_DIR

torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_transformer.py \
    --config configs/pretrain.yaml
