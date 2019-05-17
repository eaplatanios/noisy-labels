#!/usr/bin/env bash

PROJECT_ID="$(gcloud config get-value project -q)"
CONTAINER_NAME=noisy-labels-tf

# Build docker container.
docker build \
	-f docker/noisy-labels-tf.dockerfile \
	-t gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:latest .

docker push gcr.io/${PROJECT_ID}/${CONTAINER_NAME}

# Create a job.
kubectl create -f configs/job.yaml

