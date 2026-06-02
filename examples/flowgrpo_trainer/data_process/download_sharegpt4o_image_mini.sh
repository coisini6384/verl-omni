#!/bin/bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Download the ShareGPT-4o-Image-Mini editing dataset from Hugging Face and
# unpack the image archive in-place. After this script finishes the target
# directory contains the layout expected by ``qwenimageedit_sharegpt4o.py``:
#
#   <dest_dir>/
#   ├── train.jsonl         # {"prompt": "<edit instruction>", "image": "<file>"}
#   ├── test.jsonl
#   └── images/
#       ├── v2v_3165.png
#       └── ...
#
# Usage (defaults to ``$WORKSPACE/data/sharegpt4o_image_mini`` where
# ``WORKSPACE`` defaults to ``$HOME``):
#   bash examples/flowgrpo_trainer/data_process/download_sharegpt4o_image_mini.sh
#
# Or pick a custom destination:
#   bash examples/flowgrpo_trainer/data_process/download_sharegpt4o_image_mini.sh /path/to/dir

set -euo pipefail

DEST_DIR="${1:-${WORKSPACE:-$HOME}/data/sharegpt4o_image_mini}"
mkdir -p "$DEST_DIR"

# Pull train.jsonl, test.jsonl, and images.tar.gz from the HF dataset repo.
hf download coisini6384/ShareGPT-4o-Image-Mini \
    --repo-type dataset \
    --local-dir "$DEST_DIR"

# Extract images.tar.gz in-place so subsequent steps see ``<DEST_DIR>/images/``.
tar -xzvf "$DEST_DIR/images.tar.gz" -C "$DEST_DIR"

echo "Download completed. Dataset is now at: $DEST_DIR"
