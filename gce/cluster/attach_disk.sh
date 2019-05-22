#!/usr/bin/env bash

# Source cluster profile.
source "$(dirname "$0")/.profile"

# Attach disk to the gate node.
gcloud compute instances attach-disk ${CLUSTER_GATE} --disk=${CLUSTER_NAME}-disk
