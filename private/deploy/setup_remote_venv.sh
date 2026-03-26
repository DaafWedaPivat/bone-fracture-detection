#!/bin/bash

# ==========================================
# CONFIGURATION - EDIT THESE VALUES
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
    if [-d "private/generated/yolo_dataset_enhanced"]; then
      echo "Dataset folder already exists, skip download"
    else
      echo "Downloading dataset ..."
      curl -u "g9xqSHsosK7oP6N":"" -H "X-Requested-With: XMLHttpRequest" "https://cloud.vochts.de/public.php/webdav/" -o dataset.zip
      echo "Extracting dataset..."
      unzip -q -o dataset.zip -d private/dependencies/FracAtlas/
      rm dataset.zip
      echo "Dataset extracted to private/dependencies/"
    fi
else
    echo "NEXTCLOUD_DATASET_URL not configured. Skipping download."
    echo "Ensure dataset is manually placed in private/dependencies/FracAtlas"
fi

# 3. Prepare Dataset
echo "Running dataset preparation..."
export ROOT_DIR="$(pwd)/"
python3 private/make/yolo_dataset_enhanced_remote.py

# 4. Run Training
echo "Starting training"
python3 private/src/train_remote.py

echo "training finished successfully."
