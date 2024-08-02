#!usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
conda activate contactpose

if [ $# -ne 4 ]; then
    echo "Usage: ./$0 p_num intent images_dload_dir background_images_dir"
    echo "Received $# arguments instead"
    exit -1
fi

# Download
python scripts/download_data.py --p_num $1 --intent $2 --images_dload_dir $3 --type images

# Pre-process
python scripts/preprocess_images.py --p_num $1 --intent $2 --no_depth --background_images_dir $4
