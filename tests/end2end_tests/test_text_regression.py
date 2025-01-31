import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pytest
from utils import check_artifacts
from utils import get_device
from utils_text import tokenize_texts, generate_dummy_text_regression_data
from utils_text import CNNTextRegressionModel

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

device = get_device()
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path='/home/ubuntu/efs/modlee_pypi_testruns')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_samples_list = [100, 200]
modlee_trainer_list = [True, False]
recommended_model_list = [True, False]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_text_regression(num_samples, modlee_trainer, recommended_model):
    texts, targets = generate_dummy_text_regression_data(num_samples=num_samples)
    
    input_ids, attention_masks = tokenize_texts(texts, tokenizer)
    
    X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
        input_ids, attention_masks, targets, test_size=0.2, random_state=42
    )
    
    train_dataset = TextDataset(torch.tensor(X_train_ids, dtype=torch.float),torch.tensor(y_train, dtype=torch.float) )
    test_dataset = TextDataset(torch.tensor(X_test_ids, dtype=torch.float),torch.tensor(y_test, dtype=torch.float))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_dataloader.initial_tokenizer = tokenizer

    if recommended_model == True:
        recommender = modlee.recommender.from_modality_task(
            modality='text',
            task='regression'
            )
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = CNNTextRegressionModel(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    print(last_run_path)
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_text_regression(100)