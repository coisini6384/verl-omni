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
"""
Preprocess an image editing dataset to parquet format (for Qwen-Image-Edit-Plus training).

Expected input format: a dataset with columns:
  - 'instruction': text editing instruction
  - 'input_image': path or PIL image of the source image
  - 'output_image': path or PIL image of the target (ground truth)

Output format: parquet with columns expected by verl_omni data pipeline.
"""

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/dataset/image_edit/", help="Path to the raw image edit dataset.")
    parser.add_argument("--output_dir", default="~/data/image_edit", help="Directory to save preprocessed parquet.")
    parser.add_argument("--max_prompt_length", type=int, default=512)

    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    data_source = "qwen_image_edit/nft"

    dataset = datasets.load_dataset(local_dataset_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # QwenImageEditPlus prompt template matches the one used in the pipeline
    system_prompt = (
        "Describe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how the user's "
        "text instruction should alter or modify the image. Generate a new image that meets "
        "the user's requirements while maintaining consistency with the original input where "
        "appropriate."
    )
    negative_user_prompt = " "

    def make_map_fn(split):
        def process_fn(example, idx):
            instruction = example.get("instruction", example.get("prompt", ""))
            input_image = example.get("input_image", example.get("source_image", None))

            # Build the data item in verl_omni expected format
            data_item = {
                "data_source": data_source,
                "prompt": instruction,
                "negative_prompt": negative_user_prompt,
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }

            # Include source image path/data if available
            if input_image is not None:
                data_item["condition_image"] = input_image

            # Ground truth for reward computation
            if "output_image" in example:
                data_item["ground_truth"] = example["output_image"]

            return data_item

        return process_fn

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))

    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))

    print(f"Saved train ({len(train_dataset)}) and test ({len(test_dataset)}) to {output_dir}")
