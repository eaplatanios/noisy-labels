#!/usr/bin/env bash

# Source cluster profile.
source "$(dirname "$0")/.profile"

# Create a container cluster
gcloud container clusters delete ${CLUSTER_NAME} --zone ${ZONE}
