#!/bin/bash
#SBATCH --job-name="e4s-nat"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source secrets

singularity run --disable-cache --nv \
    --bind "$HF_HOME" \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "scripts/configure_setup.sh && source venv/bin/activate && \
        python -m eval4sim.naturalness data/corpora/*/*.jsonl"
