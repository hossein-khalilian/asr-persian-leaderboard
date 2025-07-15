from collections import Counter

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

# Step 1: Load the original splits
dataset_dict = load_dataset("hsekhalilian/commonvoice")

# Step 2: Concatenate all splits
full_dataset = concatenate_datasets(
    [dataset_dict["train"], dataset_dict["dev"], dataset_dict["test"]]
)

transcripts = full_dataset["normalized_transcription"]
transcript_counts = Counter(transcripts)

full_dataset = full_dataset.add_column(
    "transcript_occupancy", [transcript_counts[t] for t in transcripts]
)

full_dataset = full_dataset.sort("normalized_transcription")
full_dataset = full_dataset.sort("transcript_occupancy")

# Split
test_size = 10540
dev_size = 10540

test_data = full_dataset.select(range(test_size))
dev_data = full_dataset.select(range(test_size, test_size + dev_size))
train_data = full_dataset.select(range(test_size + dev_size, len(full_dataset)))

# Reconstruct DatasetDict
custom_dataset = DatasetDict({"train": train_data, "dev": dev_data, "test": test_data})

custom_dataset.save_to_disk("/home/user/.cache/datasets/commonvoice_sorted")
