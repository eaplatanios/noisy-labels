#!/bin/bash

CLUSTER_NAME=noisy-labels-cluster
ZONE=us-east1-b

# Create a container cluster
gcloud container clusters delete ${CLUSTER_NAME} --zone ${ZONE}

