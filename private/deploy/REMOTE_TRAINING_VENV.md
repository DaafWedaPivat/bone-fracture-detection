# Remote Training Guide (VENV Setup)

This guide is for running training on a remote server that doesn't have Docker. It uses a standard Python virtual environment (venv).

## Prerequisites
- **Python 3.x**: Ensure `python3` and `python3-venv` are installed on the server.
- **Nextcloud**:
  - **Dataset**: A zip file with public download link.

## Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/DaafWedaPivat/bone-fracture-detection.git
    cd bone-fracture-detection
    git checkout feature/remote-training-venv
    ```

2.  **Configure the Script**:
    Open `private/deploy/setup_remote_venv.sh` and fill in your Nextcloud dataset link at the top of the file:
    ```bash
    # Open with nano or vim
    nano private/deploy/setup_remote_venv.sh
    ```

3.  **Run the Script**:
    The script will handle everything else (venv creation, dependencies, download, training).
    ```bash
    ./private/deploy/setup_remote_venv.sh
    ```

## Features
- **Easy Config**: All configuration variables are clearly at the top of the script.
- **Isolation**: Uses a dedicated `.venv_remote` folder to keep the server's Python clean.
- **Local Results**: Training results are stored in `private/generated/[TRAIN_NAME]`.
