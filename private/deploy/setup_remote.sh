#!/bin/bash

# Configuration
# 1. The Nextcloud Public Link for your dataset (zip file). 
#    Should be the "Download" link. Usually ends with /download
NEXTCLOUD_DATASET_URL="${NEXTCLOUD_DATASET_URL:-}"

# 2. Nextcloud File Drop URL for results
NEXTCLOUD_UPLOAD_URL="${NEXTCLOUD_UPLOAD_URL:-}"

# 3. Training parameters
export TRAIN_NAME="yolo11n100_enhanced_remote"

# Ensure directories exist
mkdir -p private/dependencies
mkdir -p private/generated

# 1. Download and Extract Dataset
if [ -n "$NEXTCLOUD_DATASET_URL" ]; then
    echo "Downloading dataset from Nextcloud..."
    curl -L "$NEXTCLOUD_DATASET_URL" -o dataset.zip
    echo "Extracting dataset..."
    unzip -q dataset.zip -d private/dependencies/
    rm dataset.zip
    echo "Dataset extracted to private/dependencies/"
else
    echo "NEXTCLOUD_DATASET_URL not set. Skipping download. Ensure dataset is in private/dependencies/FracAtlas"
fi

# 2. Build Docker Image
echo "Building Docker image..."
docker build -t yolo-trainer -f private/deploy/Dockerfile .

# 3. Run Training (Dataset prep happens inside or before)
echo "Starting training..."
docker run --rm 
    --gpus all 
    -v "$(pwd)/private/dependencies:/usr/src/app/private/dependencies" 
    -v "$(pwd)/private/generated:/usr/src/generated" 
    -e NEXTCLOUD_UPLOAD_URL="$NEXTCLOUD_UPLOAD_URL" 
    -e TRAIN_NAME="$TRAIN_NAME" 
    yolo-trainer

echo "Remote setup finished."
