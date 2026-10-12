#!/usr/bin/env bash
# Bring up one spot L4 for a training run.
#
# Two flags here are the cost control. The run duration is capped so a job
# that hangs cannot bill overnight, and the boot disk is marked auto-delete
# so nothing outlives the instance. The termination action matters only for
# spot, where STOP would leave a stopped instance billing its disk.
#
# No --accelerator flag: G2 machine types attach their L4 automatically, and
# passing it conflicts.
#
# No service account either. The instance trains a model from files copied
# onto it and calls no Google API, so attaching credentials would only widen
# what a compromised spot VM could reach.
set -euo pipefail

PROJECT="${PROJECT:-medi-agent-490106}"
# Tried in order. L4 capacity is regional and runs out: every us-central1
# zone returned STOCKOUT for an on-demand L4 on the day this was written,
# which is also why the earlier spot instance was reclaimed. Quota is one
# GPU per region, so only one of these can succeed at a time anyway.
ZONES="${ZONE:-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east4-a us-east4-c}"
NAME="${NAME:-signalrank-l4}"
MAX_RUN="${MAX_RUN:-6h}"
# STANDARD by default, not SPOT. A spot instance running this pipeline was
# preempted at 104 minutes and, because termination deletes it, took the
# finished corpus and an almost complete mining run with it. Spot saves 40
# percent of an hourly rate that is already small, and costs the whole run
# when capacity is reclaimed. Set MODEL=SPOT to take that trade knowingly.
MODEL="${MODEL:-STANDARD}"

create_in() {
  gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$1" \
  --machine-type=g2-standard-8 \
  --provisioning-model="$MODEL" \
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
}

for zone in $ZONES; do
  echo "trying $zone"
  if create_in "$zone"; then
    echo "$zone" > .gcp-zone
    echo "running in $zone (recorded in .gcp-zone)"
    exit 0
  fi
done
echo "no zone had capacity for an L4" >&2
exit 1
