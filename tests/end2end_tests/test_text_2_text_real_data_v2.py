
### Test Function ###
import os
import modlee
import pytest
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import check_artifacts, get_device
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# Initialize device and modlee
device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path='/home/ubuntu/efs/modlee_pypi_testruns')

# Constants
BATCH_SIZE = 16
MAX_LENGTH = 50


@pytest.mark.parametrize("model_type", ["transformer", "automodel"])
@pytest.mark.parametrize("use_modlee_trainer", [True, False])
@pytest.mark.parametrize("num_samples", [100, 200])
def test_text_to_text(model_type, use_modlee_trainer, num_samples):
    input_ids, decoder_input_ids, tokenizer = load_dataset_and_tokenize(num_samples=num_samples)
    train_dataloader, val_dataloader = create_dataloaders(input_ids, decoder_input_ids)

    if model_type == "transformer":
        model = TransformerSeq2SeqModel(vocab_size=vocab_size).to(device)
    elif model_type == "automodel":
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
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
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_text_to_text("transformer", True, 100)