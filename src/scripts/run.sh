#!/usr/bin/env bash

# Create a directory for the a new experiment.
DATASET=wordsim
DATA_DIR="data"
RESULTS_DIR="results/debug"
# RESULTS_DIR="results/$(date +'%Y-%m-%d/exp_%H_%M_%S')"
mkdir -p ${RESULTS_DIR}

# Run experiment.
CUDA_VISIBLE_DEVICES="" \
    python src/scripts/run_experiment.py \
        --dataset-name ${DATASET} \
        --data-dir ${DATA_DIR} \
        --results-dir ${RESULTS_DIR} \
        --instances-emb-size 0 \
        --instances-emb-size 4 \
        --instances-emb-size 16 \
        --instances-hidden "[]" \
        --instances-hidden "[16, 16]" \
        --instances-hidden "[16, 16, 16, 16]" \
        --predictors-emb-size 4 \
        --predictors-emb-size 16 \
        --predictors-hidden "[]" \
        --q-latent-size 1 \
        --gamma 0.25 \
        --optimizer "amsgrad" \
        --batch-size 1024 \
        --max-em-iters 10 \
        --max-m-steps 1000 \
        --max-marginal-steps 1000 \
        --lambda-entropy 0. \
        --use-soft-y-hat \
        --use-progress-bar \
        --num-proc 4 \
        --seed 42
