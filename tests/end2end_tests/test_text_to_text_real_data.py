import os
import modlee
import pytest
import lightning.pytorch as pl
from utils import check_artifacts, get_device
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from utils_text import *

# Initialize device and modlee
device = get_device()
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

# Constants
BATCH_SIZE = 16
MAX_LENGTH = 50

def load_dataset_and_tokenize(num_samples, dataset_name="wmt16", subset="ro-en", split="train[:80%]", max_length=50):
    """
    Loads a dataset and tokenizes it for text-to-text tasks.

    Args:
        num_samples (int): Number of samples to load.
        dataset_name (str): Name of the dataset to load.
        subset (str): Subset of the dataset to use.
        split (str): Data split to use.
        max_length (int): Maximum token length.

    Returns:
        tuple: input_ids, decoder_input_ids, tokenizer
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, subset, split=split)

    # Select a subset of the data
    subset = dataset.select(range(num_samples))
    texts = [item['translation']['en'] for item in subset]
    target_texts = [item['translation']['ro'] for item in subset]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    target_encodings = tokenizer(
        target_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encodings['input_ids'].to(torch.long)
    decoder_input_ids = target_encodings['input_ids'].to(torch.long)

    return input_ids, decoder_input_ids, tokenizer


def create_dataloaders(input_ids, decoder_input_ids, test_size=0.2, batch_size=16):
    """
    Splits data into training and validation sets and creates dataloaders.

    Args:
        input_ids (Tensor): Input token IDs.
        decoder_input_ids (Tensor): Decoder token IDs.
        test_size (float): Proportion of data to use as test set.
        batch_size (int): Batch size.

    Returns:
        tuple: train_dataloader, val_dataloader
    """
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        input_ids, decoder_input_ids, test_size=test_size, random_state=42
    )

    # Create datasets
    train_dataset = TensorDataset(X_train, X_train, y_train)
    val_dataset = TensorDataset(X_val, X_val, y_val)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

recommended_model_list = [True]

@pytest.mark.parametrize("model_type", ["transformer", "automodel"]) 
@pytest.mark.parametrize("use_modlee_trainer", [True, False])
@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_text_to_text(model_type, use_modlee_trainer, num_samples, recommended_model):
    input_ids, decoder_input_ids, tokenizer = load_dataset_and_tokenize(num_samples=num_samples)
    train_dataloader, val_dataloader = create_dataloaders(input_ids, decoder_input_ids)

    if recommended_model == True:
        recommender = modlee.recommender.from_modality_task(
        modality='text',
        task='texttotext', 
        vocab_size=tokenizer.vocab_size
        )

        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")

    else:
        model = initialize_model(model_type, tokenizer.vocab_size, tokenizer)
    
    if use_modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=5)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        trainer = pl.Trainer(max_epochs=5)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    #train_model(model, train_dataloader, val_dataloader, use_modlee_trainer)

    last_run_path = modlee.last_run_path()
    print(last_run_path)

    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_text_to_text("transformer", True, 100)
