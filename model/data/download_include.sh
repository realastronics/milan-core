#!/usr/bin/env bash
# Downloads and unzips the raw INCLUDE dataset videos from Zenodo (record 4010759),
# per AI4Bharat's documented download process:
# https://huggingface.co/datasets/ai4bharat/INCLUDE
#
# Usage:
#   bash model/data/download_include.sh /data/milan/include/raw
#
# Requires `jq` (JSON parsing) and `curl` — both are near-universal on Ubuntu/DGX images,
# but check first: `which jq curl`.

set -euo pipefail

TARGET_DIR="${1:-./data/include/raw}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

BASE_URL="https://zenodo.org/api/records/4010759"

echo "Fetching file listing from Zenodo..."
RESPONSE=$(curl -s "$BASE_URL")

echo "$RESPONSE" | jq -r '.files[] | .links.self + " " + .key' | while read -r file_url file_name; do
    if [ -f "$file_name" ]; then
        echo "Already have $file_name, skipping."
        continue
    fi
    echo "Downloading $file_name ..."
    curl -o "$file_name" "$file_url"
done

echo "Unzipping..."
for zip_file in *.zip; do
    [ -e "$zip_file" ] || continue  # handles the case of no zip files matched
    out_dir="${zip_file%.zip}"
    if [ -d "$out_dir" ]; then
        echo "Already unzipped $zip_file, skipping."
        continue
    fi
    unzip -q "$zip_file"
done

echo "Done. Raw INCLUDE videos are in $TARGET_DIR"
echo "Next: python model/data/build_include_manifest.py --output model/data/include_manifest.csv"
