#!/bin/bash

CLUSTER_NAME=noisy-labels-cluster
DISK_SIZE=16GB
IMAGE_TYPE=UBUNTU
MACHINE_TYPE=n1-highcpu-4
NUM_NODES=2
ZONE=us-east1-b

# Create a container cluster
gcloud container clusters create ${CLUSTER_NAME} \
	--enable-autoscaler \
	--enable-ip-alias \
	--machine-type=${MACHINE_TYPE} \
	--image-type=${IMAGE_TYPE} \
	--disk-size=${DISK_SIZE} \
	--preemptible \
	--num-nodes=${NUM_NODES} \
	--zone ${ZONE}

# Enable CLI access
gcloud container clusters get-credentials ${CLUSTER_NAME} \
	--project ${PROJECT_ID} \
	--zone ${ZONE}

