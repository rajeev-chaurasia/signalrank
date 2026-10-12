#!/usr/bin/env bash
# Delete the instance and prove nothing is left billing.
#
# The listing afterwards is the point. An instance deleted while a disk,
# snapshot or reserved address survives it goes on costing money quietly, and
# the only way to know is to look.
set -euo pipefail

PROJECT="${PROJECT:-medi-agent-490106}"
# Whichever zone provisioning actually landed in.
ZONE="${ZONE:-$(cat .gcp-zone 2>/dev/null || echo us-central1-a)}"
NAME="${NAME:-signalrank-l4}"

gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet || true

echo
echo "each of these must be empty:"
for kind in instances disks snapshots addresses; do
  printf '  %-11s ' "$kind"
  gcloud compute "$kind" list --project="$PROJECT" --format="value(name)" 2>/dev/null | tr '\n' ' '
  echo
done
printf '  %-11s ' "images"
gcloud compute images list --project="$PROJECT" --no-standard-images --format="value(name)" | tr '\n' ' '
echo
