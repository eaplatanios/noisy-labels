#!/bin/bash

# Install dependencies.
pip install -e /storage/code/noisy-ml

# Create a directory for the a new experiment.
DATA_DIR="/storage/data"
RESULTS_DIR="/storage/results/$(date +'%Y-%m-%d/exp_%H_%M_%S')"
mkdir -p ${RESULTS_DIR}

# Run experiment.
python /storage/code/noisy-ml/src/scripts/run_experiment.py \
  --dataset-name rte \
  --data-dir ${DATA_DIR} \
  --results-dir ${RESULTS_DIR} \
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

