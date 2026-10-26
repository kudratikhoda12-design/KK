#!/usr/bin/env bash
# Downloads the real PmatData dataset: "A Pressure Map Dataset for In-bed
# Posture Classification" (Baran Pouyan et al., IEEE BHI 2017), hosted
# open-access on PhysioNet, DOI 10.13026/C2W09K. No login/credentialing
# required -- Open Data Commons Attribution License v1.0.
#
# Real pressure-mat recordings of 13 (Experiment I) + 8 (Experiment II)
# human subjects lying in labeled postures, used in place of the
# request-gated original PMAP dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://physionet.org/static/published-projects/pmd/pmd-pressure-map-dataset-for-in-bed-posture-classification-1.0.0.zip"
DEST_DIR="external_data/pmd"
ZIP="external_data/pmd.zip"

if [ -d "$DEST_DIR/experiment-i" ]; then
  echo "Already downloaded at $DEST_DIR -- skipping."
  exit 0
fi

mkdir -p external_data
echo "Downloading PmatData (~107MB) from PhysioNet..."
# resume-friendly: the proxy connection can be slow, so retry with -C -
for i in $(seq 1 10); do
  curl -sSL -C - -o "$ZIP" "$URL" --max-time 300 || true
  size=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  echo "  downloaded so far: $size bytes"
  # the published zip is ~107315828 bytes; stop once stable/complete
  if [ "$size" -ge 107000000 ]; then
    break
  fi
done

mkdir -p "$DEST_DIR.tmp"
unzip -q "$ZIP" -d "$DEST_DIR.tmp"
inner=$(find "$DEST_DIR.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
mv "$inner" "$DEST_DIR"
rmdir "$DEST_DIR.tmp"
rm -f "$ZIP"
echo "Done. Dataset at $DEST_DIR"
