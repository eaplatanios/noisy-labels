#!/usr/bin/env bash

# Project parameters.
export PROJECT_ID="$(gcloud config get-value project -q)"

# Cluster parameters.
export CLUSTER_NAME=noisy-labels-cluster
export CLUSTER_GATE=gce-gate
export DISK_SIZE=16GB
export IMAGE_TYPE=UBUNTU
export MACHINE_TYPE=n1-highcpu-16
export NUM_NODES=8
export ZONE=us-east1-b
