# Remote Training Setup Guide

This guide explains how to run the bone fracture detection training on a remote Linux server using Docker and Nextcloud for automated data handling.

## Prerequisites

1.  **Remote Server**: A Linux machine with `docker` and `nvidia-container-toolkit` installed (for GPU support).
2.  **Nextcloud**:
    *   **Dataset**: Upload your `FracAtlas` dataset as a `.zip` file and create a **Public Link** (Download link).
    *   **Results**: Create a folder for results and create a **Public Link** with **File Drop** (Upload only) or **Allow upload and editing** enabled.

## Step-by-Step Instructions

### 1. Clone the Repository
Connect to your server via SSH and clone the project:
```bash
git clone https://github.com/DaafWedaPivat/bone-fracture-detection.git
cd bone-fracture-detection
git checkout feature/remote-training-docker
```

### 2. Configure Environment Variables
Set your Nextcloud links as environment variables. 

*   `NEXTCLOUD_DATASET_URL`: The direct download link to your dataset zip.
*   `NEXTCLOUD_UPLOAD_URL`: The WebDAV endpoint for your result folder. 
    *   *Note: For Nextcloud File Drops, the WebDAV URL is usually `https://your-nextcloud.com/public.php/dav/SHARE_TOKEN`.*

```bash
export NEXTCLOUD_DATASET_URL="https://your-nextcloud.com/index.php/s/DATASET_TOKEN/download"
export NEXTCLOUD_UPLOAD_URL="https://your-nextcloud.com/public.php/dav/UPLOAD_TOKEN"
```

### 3. Run the Setup Script
Execute the provided setup script. This script will:
1. Download and extract the dataset.
2. Build the Docker image with all dependencies.
3. Start the training process inside a container.
4. Archive all results (`.tar.gz`) and upload them to your Nextcloud.

```bash
./private/deploy/setup_remote.sh
```

## Customization

You can change training parameters by setting additional environment variables before running the script:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TRAIN_NAME` | Name of the training run | `yolo11n100_enhanced_remote` |
| `EPOCHS` | Number of training epochs | `100` |
| `IMGSZ` | Image size | `640` |

Example:
```bash
export EPOCHS=50
./private/deploy/setup_remote.sh
```

## Troubleshooting

*   **No GPU**: If the server doesn't have an NVIDIA GPU, remove the `--gpus all` flag from `private/deploy/run_remote.sh` or `private/deploy/setup_remote.sh`.
*   **Upload Fails**: Ensure your `NEXTCLOUD_UPLOAD_URL` is the correct WebDAV path. You can test it manually with:
    `curl -T some_file.txt "https://your-nextcloud.com/public.php/dav/TOKEN/some_file.txt"`
