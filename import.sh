#!/bin/bash

# Set proxy environment variables
export http_proxy="http://proxy:80"
export https_proxy="http://proxy:80"

# Create a directory for the model
MODEL_DIR="./models"
mkdir -p $MODEL_DIR

# Download the pretrained model with proxy support
MODEL_URL="https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth"
MODEL_PATH="$MODEL_DIR/hiera_base_224.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model with proxy..."
    wget -e use_proxy=yes -e http_proxy=$http_proxy -e https_proxy=$https_proxy -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Model already downloaded."
fi

echo "Model saved at: $MODEL_PATH"
