#!/usr/bin/env bash
# Bring up one spot L4 for a training run.
#
# Three flags here are the cost control, and none of them is optional. Spot
# instances default to STOP on termination, and a stopped instance still bills
# its disk, so the termination action is set to DELETE. The run duration is
# capped so a job that hangs cannot bill overnight. The boot disk is marked
# auto-delete so nothing outlives the instance.
#
# No --accelerator flag: G2 machine types attach their L4 automatically, and
# passing it conflicts.
#
# No service account either. The instance trains a model from files copied
# onto it and calls no Google API, so attaching credentials would only widen
# what a compromised spot VM could reach.
set -euo pipefail

PROJECT="${PROJECT:-medi-agent-490106}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-signalrank-l4}"
MAX_RUN="${MAX_RUN:-6h}"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --max-run-duration="$MAX_RUN" \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-balanced \
  --boot-disk-auto-delete \
  --boot-disk-device-name="$NAME" \
  --metadata="install-nvidia-driver=True" \
  --no-service-account \
  --no-scopes \
  --labels=purpose=signalrank
