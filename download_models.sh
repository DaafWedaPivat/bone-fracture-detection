#!/bin/bash

# Exit on error
set -e



# Ensure directories exist
mkdir -p private/dependencies/models

# Download and Extract trained models
echo "Downloading models ..."

curl -u "wCzg5iN7dsjFZzM":"" -H "X-Requested-With: XMLHttpRequest" "https://cloud.vochts.de/public.php/webdav/" -o models.zip
echo "Extracting models..."
unzip -q -o models.zip -d private/dependencies/models/
rm models.zip
echo "models extracted to private/dependecies/models"
