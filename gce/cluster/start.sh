#!/usr/bin/env bash

# Source cluster profile.
source "$(dirname "$0")/.profile"

# Create a container cluster
gcloud container clusters create ${CLUSTER_NAME} \
	--enable-ip-alias \
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

