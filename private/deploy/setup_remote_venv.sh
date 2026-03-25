#!/bin/bash

# ==========================================
# CONFIGURATION - EDIT THESE VALUES
# ==========================================
# 1. The Nextcloud Public Link for your dataset zip (e.g., .../download)
NEXTCLOUD_DATASET_URL="https://your-nextcloud.com/index.php/s/DATASET_TOKEN/download"

# 2. Training parameters
TRAIN_NAME="yolo11n100_enhanced_remote_venv"
EPOCHS=100
IMGSZ=640
# ==========================================

# Exit on error
set -e

# Ensure directories exist
mkdir -p private/dependencies
mkdir -p private/generated

# 1. Create Virtual Environment and Install Dependencies
echo "Setting up Python virtual environment..."
if [ ! -d ".venv_remote" ]; then
    python3 -m venv .venv_remote
fi
source .venv_remote/bin/activate

echo "Installing/Updating dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download and Extract Dataset
if [[ $NEXTCLOUD_DATASET_URL != *"your-nextcloud.com"* ]]; then
    echo "Downloading dataset from Nextcloud..."
    curl -L "$NEXTCLOUD_DATASET_URL" -o dataset.zip
    echo "Extracting dataset..."
    unzip -q -o dataset.zip -d private/dependencies/
    rm dataset.zip
    echo "Dataset extracted to private/dependencies/"
else
    echo "NEXTCLOUD_DATASET_URL not configured. Skipping download."
    echo "Ensure dataset is manually placed in private/dependencies/FracAtlas"
fi

# 3. Prepare Dataset
echo "Running dataset preparation..."
export ROOT_DIR="$(pwd)/"
python3 private/make/yolo_dataset_enhanced_remote.py

# 4. Run Training
echo "Starting training: $TRAIN_NAME"
# Pass parameters to the training script via environment variables
export TRAIN_NAME="$TRAIN_NAME"
export EPOCHS="$EPOCHS"
export IMGSZ="$IMGSZ"
python3 private/src/train_remote.py

echo "Remote VENV training finished successfully."
echo "Results are saved in: private/generated/$TRAIN_NAME"
