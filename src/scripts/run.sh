#!/usr/bin/env bash

# Create a directory for the a new experiment.
exp_dir="results/$(date +'%Y-%m-%d/exp_%H_%M_%S')"
mkdir -p ${exp_dir}

# Run experiment.
CUDA_VISIBLE_DEVICES="" \
    python src/scripts/run_experiment.py \
        --dataset-name wordsim \
        --data-dir data \
        --results-dir ${exp_dir} \
        --instances-emb-size 4 \
        --instances-emb-size 0 \
        --instances-hidden "[]" \
        --instances-hidden "[16, 16]" \
        --predictors-emb-size 4 \
        --predictors-emb-size 16 \
        --predictors-hidden "[]" \
        --q-latent-size 1 \
        --gamma 0.50 \
        --gamma 0.75 \
        --gamma 1.00 \
        --num-proc 4 \
        --seed 42
