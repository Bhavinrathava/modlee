import os
import torch
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from utils import check_artifacts
from utils import get_device
import pytest
from transformers import AutoTokenizer

import random

device = get_device()
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path='/home/ubuntu/efs/modlee_pypi_testruns')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def generate_synthetic_data(num_samples=100):
    inputs = [f"Input sentence {i} {' '.join(random.choices(['random', 'words', 'to', 'add'], k=3))}" for i in range(num_samples)]
    outputs = [f"Output sentence {i} {' '.join(random.choices(['generated', 'output', 'text'], k=3))}" for i in range(num_samples)]
    return inputs, outputs

class SimpleTextToTextModel(modlee.model.TextTexttotextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, 128)  # Modified to process per timestep
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size)  # Predict per token
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # Embed input tokens
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)  # Shape: (batch_size, max_length, embed_dim)

        # Process each token independently
        x = torch.nn.functional.relu(self.fc1(embedded))  # Shape: (batch_size, max_length, 128)
        x = torch.nn.functional.relu(self.fc2(x))         # Shape: (batch_size, max_length, 128)
        x = self.fc3(x)                                   # Shape: (batch_size, max_length, vocab_size)

        return x  # Shape: (batch_size, max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ExtendedTextToTextModel(modlee.model.TextTexttotextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, 256)  # Modified to process per timestep
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size)  # Predict per token
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # Embed input tokens
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)  # Shape: (batch_size, max_length, embed_dim)

        # Process each token independently
        x = torch.nn.functional.relu(self.fc1(embedded))  # Shape: (batch_size, max_length, 256)
        x = torch.nn.functional.relu(self.fc2(x))         # Shape: (batch_size, max_length, 128)
        x = self.fc3(x)                                   # Shape: (batch_size, max_length, vocab_size)

        return x  # Shape: (batch_size, max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class SimplifiedTextToTextModel(modlee.model.TextTexttotextModleeModel):
    def __init__(self, vocab_size, embed_dim=30, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, 64)  # Modified to process per timestep
        self.fc2 = torch.nn.Linear(64, vocab_size)  # Predict per token
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # Embed input tokens
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)  # Shape: (batch_size, max_length, embed_dim)

        # Process each token independently
        x = torch.nn.functional.relu(self.fc1(embedded))  # Shape: (batch_size, max_length, 64)
        x = self.fc2(x)                                   # Shape: (batch_size, max_length, vocab_size)

        return x  # Shape: (batch_size, max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ResidualTextToTextModel(modlee.model.TextTexttotextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, 128)  # Modified to process per timestep
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size)  # Predict per token
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # Embed input tokens
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)  # Shape: (batch_size, max_length, embed_dim)

        # Process each token independently
        x = torch.nn.functional.relu(self.fc1(embedded))  # Shape: (batch_size, max_length, 128)
        residual = x  # Residual connection
        x = torch.nn.functional.relu(self.fc2(x)) + residual  # Shape: (batch_size, max_length, 128)
        x = self.fc3(x)  # Shape: (batch_size, max_length, vocab_size)

        return x  # Shape: (batch_size, max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)  # Shape: (batch_size, max_length, vocab_size)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(  # Ignore padding token
            preds.view(-1, self.vocab_size), 
            targets.view(-1)
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

recommended_model_list = [True, False]

@pytest.mark.parametrize("modlee_trainer", [False, True])
@pytest.mark.parametrize("num_samples", [100])
@pytest.mark.parametrize("model_class", [
    SimpleTextToTextModel,
    ExtendedTextToTextModel,
    SimplifiedTextToTextModel,
    ResidualTextToTextModel
])
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_text_to_text(modlee_trainer, num_samples, model_class, recommended_model):
    inputs, outputs = generate_synthetic_data(num_samples=num_samples)

    vocab = {word: idx for idx, word in enumerate(set(" ".join(inputs + outputs).split()))}
    
    pad_token_id = 0  # Define padding token ID
    max_length = 20  # Fixed sequence length

    input_ids = [
        torch.tensor(
            [vocab[word] for word in text.split()] + [pad_token_id] * (max_length - len(text.split())),
            dtype=torch.float
        )[:max_length] for text in inputs
    ]

    # Generate output_ids as token indices and flatten them
    output_ids = [
        torch.tensor(
            [vocab[word] for word in text.split()] + [pad_token_id] * (max_length - len(text.split())),
            dtype=torch.long
        )[:max_length] for text in outputs
    ]

    X_train, X_test, y_train, y_test = train_test_split(input_ids, output_ids, test_size=0.2)

    class TextDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader.initial_tokenizer = tokenizer

    if recommended_model == True:
        recommender = modlee.recommender.from_modality_task(
            modality='text',
            task='texttotext', 
            vocab_size=len(vocab)
            )

        recommender.fit(train_dataloader)
        model = recommender.model
        print(f"\nRecommended model: \n{model}")
    else:
        model = model_class(vocab_size=len(vocab), max_length=max_length).to(device)
    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    print(last_run_path)
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_text_to_text(False, 100, SimpleTextToTextModel)