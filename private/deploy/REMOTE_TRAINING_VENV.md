# Remote Training Guide (VENV Setup)

This guide is for running training on a remote server that doesn't have Docker. It uses a standard Python virtual environment (venv).

## Prerequisites
- **Python 3.x**: Ensure `python3` and `python3-venv` are installed on the server.
- **Tools**: Ensure `curl` and `unzip` are installed.

## Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/DaafWedaPivat/bone-fracture-detection.git
    cd bone-fracture-detection
    git checkout feature/remote-training-venv
    ```

2.  **Run the Script**:
    The script handles everything (venv creation, dependencies, dataset download, training).
    ```bash
    ./private/deploy/setup_remote_venv.sh
    ```

## Features
- **Hardcoded Download**: The dataset is automatically downloaded using the pre-configured token.
- **Easy Config**: Training parameters (epochs, name) are at the top of the script.
- **Isolation**: Uses a dedicated `.venv_remote` folder to keep the server's Python clean.
- **Local Results**: Training results are stored in `private/generated/[TRAIN_NAME]`.