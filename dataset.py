import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


DATASET_PATH = Path(__file__).parent / "translation_dataset.json"


class TranslationDataset(Dataset):
    """
    Dataset for English -> French sentence pairs.

    Each item contains:
        source: English sentence
        target: French sentence
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        return {
            "source": item["source"],
            "target": item["target"],
        }


def load_translation_data(path=DATASET_PATH):
    """
    Load the complete translation dataset from JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Dataset JSON must contain a list of objects."
        )

    for item in data:
        if "source" not in item or "target" not in item:
            raise ValueError(
                "Every dataset item must contain 'source' and 'target'."
            )

    return data


def split_dataset(
    data,
    train_ratio=0.8,
    validation_ratio=0.1,
    seed=7,
):
    """
    Split data into:
        80% training
        10% validation
        10% testing
    """

    data = list(data)

    random_generator = random.Random(seed)
    random_generator.shuffle(data)

    total = len(data)

    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    test_data = data[validation_end:]

    return train_data, validation_data, test_data


def pad_sequences(
    sequences,
    pad_id,
):
    """
    Pad a list of token ID sequences to the same length.

    Returns:
        tensor of shape:
        (batch_size, max_sequence_length)
    """

    max_length = max(len(sequence) for sequence in sequences)

    padded_sequences = []

    for sequence in sequences:
        padding_length = max_length - len(sequence)

        padded_sequence = (
            sequence
            + [pad_id] * padding_length
        )

        padded_sequences.append(padded_sequence)

    return torch.tensor(
        padded_sequences,
        dtype=torch.long,
    )


def create_collate_fn(
    source_tokenizer,
    target_tokenizer,
):
    """
    Creates the function used by DataLoader to build batches.
    """

    def collate_fn(batch):

        source_sequences = [
            source_tokenizer.encode(item["source"])
            for item in batch
        ]

        target_sequences = [
            target_tokenizer.encode(item["target"])
            for item in batch
        ]

        source_tokens = pad_sequences(
            source_sequences,
            source_tokenizer.get_pad_id(),
        )

        target_tokens = pad_sequences(
            target_sequences,
            target_tokenizer.get_pad_id(),
        )

        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "source_text": [
                item["source"] for item in batch
            ],
            "target_text": [
                item["target"] for item in batch
            ],
        }

    return collate_fn