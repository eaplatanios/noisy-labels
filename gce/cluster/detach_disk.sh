#!/usr/bin/env bash

# Source cluster profile.
source "$(dirname "$0")/.profile"

# Detach disk to the gate node.
gcloud compute instances detach-disk ${CLUSTER_GATE} --disk=${CLUSTER_NAME}-disk
