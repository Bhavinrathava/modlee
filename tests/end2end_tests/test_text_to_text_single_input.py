# import os
# import torch
# import modlee
# import lightning.pytorch as pl
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from utils import check_artifacts
# from transformers import AutoTokenizer
# from utils import get_device
# import pytest
# import random

# device = get_device()
# modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def generate_synthetic_data(num_samples=100):
#     inputs = [f"Input sentence {i} {' '.join(random.choices(['random', 'words', 'to', 'add'], k=3))}" for i in range(num_samples)]
#     outputs = [f"Output sentence {i} {' '.join(random.choices(['generated', 'output', 'text'], k=3))}" for i in range(num_samples)]
#     return inputs, outputs

# class SimpleTextToTextModel(modlee.model.TextTextToTextModleeModel):
#     def __init__(self, vocab_size, embed_dim=50, max_length=20):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
#         self.fc1 = torch.nn.Linear(embed_dim * max_length, 128)
#         self.fc2 = torch.nn.Linear(128, 128)
#         self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
#         self.max_length = max_length
#         self.vocab_size = vocab_size

#     def forward(self, input_ids):
#         embedded = self.embedding(input_ids)
#         embedded = embedded.flatten(start_dim=1)
#         x = torch.nn.functional.relu(self.fc1(embedded))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x.view(-1, self.max_length, self.vocab_size)

#     def training_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

# class ExtendedTextToTextModel(modlee.model.TextTextToTextModleeModel):
#     def __init__(self, vocab_size, embed_dim=50, max_length=20):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
#         self.fc1 = torch.nn.Linear(embed_dim * max_length, 256)
#         self.fc2 = torch.nn.Linear(256, 128)
#         self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
#         self.max_length = max_length
#         self.vocab_size = vocab_size

#     def forward(self, input_ids):
#         embedded = self.embedding(input_ids)
#         embedded = embedded.flatten(start_dim=1)
#         x = torch.nn.functional.relu(self.fc1(embedded))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x.view(-1, self.max_length, self.vocab_size)

#     def training_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

# class SimplifiedTextToTextModel(modlee.model.TextTextToTextModleeModel):
#     def __init__(self, vocab_size, embed_dim=30, max_length=20):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
#         self.fc1 = torch.nn.Linear(embed_dim * max_length, 64)
#         self.fc2 = torch.nn.Linear(64, vocab_size * max_length)
#         self.max_length = max_length
#         self.vocab_size = vocab_size

#     def forward(self, input_ids):
#         embedded = self.embedding(input_ids)
#         embedded = embedded.flatten(start_dim=1)
#         x = torch.nn.functional.relu(self.fc1(embedded))
#         x = self.fc2(x)
#         return x.view(-1, self.max_length, self.vocab_size)

#     def training_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

# class ResidualTextToTextModel(modlee.model.TextTextToTextModleeModel):
#     def __init__(self, vocab_size, embed_dim=50, max_length=20):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
#         self.fc1 = torch.nn.Linear(embed_dim * max_length, 128)
#         self.fc2 = torch.nn.Linear(128, 128)
#         self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
#         self.max_length = max_length
#         self.vocab_size = vocab_size

#     def forward(self, input_ids):
#         embedded = self.embedding(input_ids)
#         embedded = embedded.flatten(start_dim=1)
#         x = torch.nn.functional.relu(self.fc1(embedded))
#         residual = x
#         x = torch.nn.functional.relu(self.fc2(x)) + residual
#         x = self.fc3(x)
#         return x.view(-1, self.max_length, self.vocab_size)

#     def training_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids, targets = batch
#         preds = self.forward(input_ids)
#         loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

# @pytest.mark.parametrize("modlee_trainer", [True, False])
# @pytest.mark.parametrize("model_class", [SimpleTextToTextModel, ExtendedTextToTextModel, SimplifiedTextToTextModel, ResidualTextToTextModel])
# def test_text_to_text(modlee_trainer, model_class):
#     # Generate synthetic data
#     inputs, outputs = generate_synthetic_data(num_samples=100)

#     # Create a simple vocabulary
#     vocab = {word: idx for idx, word in enumerate(set(" ".join(inputs + outputs).split()))}
#     reverse_vocab = {idx: word for word, idx in vocab.items()}

#     # Tokenize data
#     input_ids = [torch.tensor([vocab[word] for word in text.split()]) for text in inputs]
#     output_ids = [torch.tensor([vocab[word] for word in text.split()]) for text in outputs]

#     # Pad sequences
#     max_length = max(len(seq) for seq in input_ids + output_ids)
#     input_ids = [torch.nn.functional.pad(seq, (0, max_length - len(seq)), value=0) for seq in input_ids]
#     output_ids = [torch.nn.functional.pad(seq, (0, max_length - len(seq)), value=0) for seq in output_ids]

#     X_train, X_test, y_train, y_test = train_test_split(input_ids, output_ids, test_size=0.2)

#     # Prepare datasets
#     class TextDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.y)

#         def __getitem__(self, idx):
#             return self.X[idx], self.y[idx]

#     # train_dataset = TextDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
#     # test_dataset = TextDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
#     train_dataset = TextDataset(torch.stack(X_train), torch.stack(y_train))
#     test_dataset = TextDataset(torch.stack(X_test), torch.stack(y_test))

#     train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#     train_dataloader.initial_tokenizer = tokenizer

#     model = model_class(vocab_size=len(vocab), max_length=max_length).to(device)

#     if modlee_trainer:
#         trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
#         trainer.fit(
#             model=model,
#             train_dataloaders=train_dataloader,
#             val_dataloaders=test_dataloader
#         )
#     else:
#         with modlee.start_run() as run:
#             trainer = pl.Trainer(max_epochs=1)
#             trainer.fit(
#                 model=model,
#                 train_dataloaders=train_dataloader,
#                 val_dataloaders=test_dataloader
#             )

#     last_run_path = modlee.last_run_path()
#     artifacts_path = os.path.join(last_run_path, 'artifacts')
#     print(last_run_path)
#     check_artifacts(artifacts_path)



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
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def generate_synthetic_data(num_samples=100):
    inputs = [f"Input sentence {i} {' '.join(random.choices(['random', 'words', 'to', 'add'], k=3))}" for i in range(num_samples)]
    outputs = [f"Output sentence {i} {' '.join(random.choices(['generated', 'output', 'text'], k=3))}" for i in range(num_samples)]
    return inputs, outputs

class SimpleTextToTextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * max_length, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(embedded))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.max_length, self.vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ExtendedTextToTextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * max_length, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(embedded))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.max_length, self.vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class SimplifiedTextToTextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, vocab_size, embed_dim=30, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * max_length, 64)
        self.fc2 = torch.nn.Linear(64, vocab_size * max_length)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(embedded))
        x = self.fc2(x)
        return x.view(-1, self.max_length, self.vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ResidualTextToTextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, vocab_size, embed_dim=50, max_length=20):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * max_length, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, vocab_size * max_length)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(embedded))
        residual = x
        x = torch.nn.functional.relu(self.fc2(x)) + residual
        x = self.fc3(x)
        return x.view(-1, self.max_length, self.vocab_size)

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        preds = self.forward(input_ids)
        loss = torch.nn.CrossEntropyLoss()(preds.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@pytest.mark.parametrize("modlee_trainer", [True, False])
@pytest.mark.parametrize("model_class", [
    SimpleTextToTextModel,
    ExtendedTextToTextModel,
    SimplifiedTextToTextModel,
    ResidualTextToTextModel
])
def test_text_to_text(modlee_trainer, model_class):
    # Generate synthetic data
    inputs, outputs = generate_synthetic_data(num_samples=100)

    # Create a simple vocabulary
    vocab = {word: idx for idx, word in enumerate(set(" ".join(inputs + outputs).split()))}
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    # Tokenize data
    input_ids = [torch.tensor([vocab[word] for word in text.split()]) for text in inputs]
    output_ids = [torch.tensor([vocab[word] for word in text.split()]) for text in outputs]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(input_ids, output_ids, test_size=0.2)

    # Prepare datasets
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

    # Use batch_size=1 for single input-output processing
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader.initial_tokenizer = tokenizer

    model = model_class(vocab_size=len(vocab), max_length=max(len(seq) for seq in input_ids)).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=50)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=50)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    print(last_run_path)
    check_artifacts(artifacts_path)


