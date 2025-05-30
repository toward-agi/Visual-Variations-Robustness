import os
import json
import random
from datasets import load_from_disk, DatasetDict, Dataset
from tqdm import tqdm
from PIL import Image

def extract_and_split_datasets(input_dir, output_dir, output_images_dir, output_json, remaining_data_dir):
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(remaining_data_dir, exist_ok=True)

    merged_data = []

    # Iterate over all datasets in the directory
    for dataset_name in tqdm(os.listdir(input_dir)):
        dataset_path = os.path.join(input_dir, dataset_name)
        if os.path.isdir(dataset_path):
            # Load dataset
            dataset = load_from_disk(dataset_path)
            test_set = dataset['test']

            # Extract 10% of the dataset
            num_samples = max(1, int(0.1 * len(test_set)))
            sampled_data = test_set.shuffle(seed=42).select(range(num_samples))
            
            # Remaining 90% of the data
            remaining_data = test_set.select(range(num_samples, len(test_set)))

            # Save remaining data to the new directory
            remaining_dataset_path = os.path.join(remaining_data_dir, dataset_name)
            DatasetDict({"test": remaining_data}).save_to_disk(remaining_dataset_path)

            # Process each data entry
            for idx, sample in enumerate(sampled_data):
                # Save the image to the output directory
                image_path = os.path.join(output_images_dir, f"{dataset_name}_{idx}.png")
                sample['image'].save(image_path)

                # Create a new entry for the merged dataset
                merged_data.append({
                    "messages": [
                        {
                            "content": "<image>This is a coordinate plot with a single point. Provide the coordinate in the format (x,) for 1D, (x, y) for 2D, or (x, y, z) for 3D." ,
                            "role": "user"
                        },
                        {
                            "content": sample['coordinate'],  # Use the 'path' field as the assistant's response
                            "role": "assistant"
                        }
                    ],
                    "images": [
                        os.path.abspath(image_path)  # Absolute path for the JSON output
                    ]
                })

    # Save the merged dataset as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_directory = "coordinate_dataset"  # Directory containing the original datasets
    output_directory = "coordinate_merged_dataset"  # Directory for the merged dataset
    output_images_directory = os.path.join(output_directory, "images")  # Directory for images
    output_json_file = os.path.join(output_directory, "merged_dataset.json")  # JSON output file
    remaining_directory = "coordinate_remaining_dataset"  # Directory for the remaining 90% data

    extract_and_split_datasets(input_directory, output_directory, output_images_directory, output_json_file, remaining_directory)
