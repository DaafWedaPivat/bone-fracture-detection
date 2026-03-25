#!/bin/bash

# Configuration
# Replace these with your actual values if needed or pass as environment variables
DOCKER_IMAGE_NAME="yolo-trainer"
# Nextcloud File Drop URL (public upload link).
# Example: https://nextcloud.example.com/public.php/dav/
# Note: Ensure it's the WebDAV compatible URL or the public.php/dav/ endpoint.
NEXTCLOUD_UPLOAD_URL="${NEXTCLOUD_UPLOAD_URL:-}"
DATASET_PATH_HOST="$(pwd)/private/generated" # Assumes you have the dataset in private/generated

# 1. Build the Docker Image
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE_NAME -f private/deploy/Dockerfile .

# 2. Run the Container
# We mount the host's generated/ directory to /usr/src/generated/ in the container
# This provides the dataset and allows the container to save training results back to the host.
echo "Starting training container..."
docker run --rm
    --gpus all
    -v "$DATASET_PATH_HOST:/usr/src/generated"
    -e NEXTCLOUD_UPLOAD_URL="$NEXTCLOUD_UPLOAD_URL"
    -e TRAIN_NAME="yolo11n100_enhanced_remote"
    $DOCKER_IMAGE_NAME

echo "Container execution finished."
