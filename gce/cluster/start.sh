#!/bin/bash

PROJECT_ID="$(gcloud config get-value project -q)"
CLUSTER_NAME=noisy-labels-cluster
DISK_SIZE=16GB
IMAGE_TYPE=UBUNTU
MACHINE_TYPE=n1-highcpu-4
MAX_NODES=8
MIN_NODES=0
NUM_NODES=5
ZONE=us-east1-b

# Create a container cluster
gcloud container clusters create ${CLUSTER_NAME} \
	--enable-autoscaling \
	--enable-ip-alias \
	--max-nodes=${MAX_NODES} \
	--min-nodes=${MIN_NODES} \
	--num-nodes=${NUM_NODES} \
	--machine-type=${MACHINE_TYPE} \
	--image-type=${IMAGE_TYPE} \
	--disk-size=${DISK_SIZE} \
	--preemptible \
	--zone ${ZONE}

# Enable CLI access
gcloud container clusters get-credentials ${CLUSTER_NAME} \
	--project ${PROJECT_ID} \
	--zone ${ZONE}

# Release disk
gcloud compute instances detach-disk instance-1 --disk=${CLUSTER_NAME}-disk

# Create persistent volumes
kubectl apply -f configs/ext4-pv.yaml

