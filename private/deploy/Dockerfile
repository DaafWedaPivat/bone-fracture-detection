FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create result directory
RUN mkdir -p /usr/src/generated

# Copy source code and necessary files
COPY private/src/ ./src/
COPY private/make/ ./make/

# Create a shell script to run training and upload results
RUN echo '#!/bin/bash\n\
# Set default training name\n\
export TRAIN_NAME=${TRAIN_NAME:-yolo11n100_enhanced}\n\
\n\
# 1. Prepare dataset from mounted dependencies folder\n\
echo "Running dataset preparation..."\n\
export ROOT_DIR="/usr/src/app/"\n\
python3 make/yolo_dataset_enhanced_remote.py\n\
\n\
# 2. Run training using the dedicated remote script\n\
echo "Starting training: $TRAIN_NAME"\n\
python3 src/train_remote.py\n\
\n\
# 3. Archive and Upload the whole results folder to Nextcloud File Drop if URL is provided\n\
if [ -n "$NEXTCLOUD_UPLOAD_URL" ]; then\n\
  echo "Archiving results..."\n\
  # Create a zip of the entire training run folder\n\
  cd /usr/src/generated/\n\
  tar -czf "${TRAIN_NAME}_results.tar.gz" "$TRAIN_NAME"\n\
  \n\
  echo "Uploading archived results to Nextcloud File Drop..."\n\
  # Nextcloud Public Link Upload uses the URL directly (WebDAV)\n\
  # Note: The user should provide the WebDAV endpoint for the file drop share.\n\
  # Example: https://nextcloud.example.com/public.php/dav/S3P4v_token_here\n\
  curl -T "${TRAIN_NAME}_results.tar.gz" "$NEXTCLOUD_UPLOAD_URL/${TRAIN_NAME}_results.tar.gz"\n\
  echo "Upload complete!"\n\
fi' > entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]