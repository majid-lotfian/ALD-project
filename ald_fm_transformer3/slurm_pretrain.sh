#!/bin/bash
#SBATCH --job-name=ald_pretrain_v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 process per GPU (4 A100s per node)
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18           # 72 CPUs / 4 GPUs = 18 per task
#SBATCH --mem=460G                   # node has 480G; leave ~20G for OS
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=majid.lotfian@gmail.com

mkdir -p logs

# ---- environment ----
module purge
module load 2024

# Conda is initialised in .bashrc; use the shell hook so activate works
eval "$(conda shell.bash hook)"
conda activate ald

# ---- distributed setup ----
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "MASTER_ADDR=$MASTER_ADDR  WORLD_SIZE=$WORLD_SIZE  GPUS=$SLURM_GPUS_ON_NODE"
echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"

# Abort early if we're not in the ald env (wrong torch = cryptic errors later)
python -c "import torch; assert hasattr(torch.amp, 'GradScaler'), 'check failed'" || {
    echo "ERROR: torch.amp.GradScaler not found. Ensure the 'ald' conda env activated correctly."
    exit 1
}

# ---- launch ----
cd $SLURM_SUBMIT_DIR

# Dry-run: single rank, 2 files, 2 steps — catches import/config/runtime bugs in ~15s
# before committing to the full 24-hour job.
echo "=== DRY-RUN START (single rank, 2 files, 2 steps) ==="
python pretrain_transformer.py \
    --config configs/pretrain.yaml \
    --dry-run \
  || { echo "DRY-RUN FAILED — aborting job. Check the error above."; exit 1; }
echo "=== DRY-RUN PASSED — launching full torchrun ==="

torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_transformer.py \
    --config configs/pretrain.yaml
