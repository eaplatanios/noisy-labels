#!/usr/bin/env bash

# Install dependencies.
pip install -e /storage/code/noisy-ml

# Create a directory for the a new experiment.
DATASET=age
DATA_DIR="/storage/data"
RESULTS_DIR="/storage/results/${DATASET}"
mkdir -p ${RESULTS_DIR}

# Run experiment.
python /storage/code/noisy-ml/src/scripts/run_experiment.py \
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
    --gamma 0.00 \
    --gamma 0.25 \
    --gamma 0.50 \
    --gamma 0.75 \
    --gamma 1.00 \
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
