#!/bin/bash
#SBATCH --job-name="e4s-gen"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source secrets

declare -A MODELS
MODELS["qwen3_1.7b"]="Qwen/Qwen3-1.7B"
MODELS["qwen3_4b"]="Qwen/Qwen3-4B-Instruct-2507"
MODELS["qwen3_14b"]="Qwen/Qwen3-14B"
MODELS["qwen3_30b"]="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODELS["gemma3_1b"]="google/gemma-3-1b-it"
MODELS["gemma3_4b"]="google/gemma-3-4b-it"
MODELS["gemma3_12b"]="google/gemma-3-12b-it"
MODELS["gemma3_27b"]="google/gemma-3-27b-it"

for name in "${!MODELS[@]}"; do
    for seed in 1 2 3 4 5; do
        singularity run --disable-cache --nv \
            --bind "$HF_HOME" \
            --pwd "$PWD" \
            $SIF \
            /bin/bash -c "scripts/configure_setup.sh && source venv/bin/activate && \
                python -m eval4sim.generate --model ${MODELS[$name]} --personas data/corpora/spc/test.jsonl \
                    --seed $seed --output data/corpora/$name/seed$seed.jsonl"
    done
done
