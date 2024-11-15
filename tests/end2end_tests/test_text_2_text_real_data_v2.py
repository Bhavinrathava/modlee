import torch
import torch.nn as nn
import torch.optim as optim
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset  
from transformers import AutoTokenizer
import pytest
from utils import check_artifacts, get_device

device = get_device()

modlee.init(api_key='MsvVailphdUF2pcwbRRqrhx7ibxGC05W', run_path='/home/ubuntu/efs/modlee_pypi_testruns')

def load_real_data():
    dataset = load_dataset("wmt16", "ro-en", split='train[:80%]')
    print(f"Dataset structure: {dataset}")
    return dataset

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Custom Transformer-based Seq2Seq model
class TransformerSeq2SeqModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, max_length=50):
        super(TransformerSeq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(d_model, max_length), requires_grad=False)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.d_model = d_model

    def forward(self, input_ids, decoder_input_ids=None):

        if decoder_input_ids == None:
            decoder_input_ids = input_ids

        if isinstance(input_ids, list) and len(input_ids)== 2:
            (input_ids, decoder_input_ids) = input_ids

        # Add positional encoding to embeddings
        src = self.embedding(input_ids) * (self.d_model ** 0.5) + self.positional_encoding[:input_ids.size(1), :]
        tgt = self.embedding(decoder_input_ids) * (self.d_model ** 0.5) + self.positional_encoding[:decoder_input_ids.size(1), :]

        src = src.permute(1, 0, 2)  # Transformer expects sequence as first dimension (seq_len, batch, d_model)
        tgt = tgt.permute(1, 0, 2)

        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory)
        
        logits = self.fc_out(output.permute(1, 0, 2))  # Revert back to (batch, seq_len, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch
        logits = self(input_ids, decoder_input_ids)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch
        logits = self(input_ids, decoder_input_ids)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-5)
    
    @staticmethod
    def _generate_positional_encoding(d_model, max_length):
        pos = torch.arange(0, max_length).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle_rates = 1 / torch.pow(10000, (i.float() / d_model))
        pos_enc = torch.zeros(max_length, d_model)
        pos_enc[:, 0::2] = torch.sin(pos * angle_rates)
        pos_enc[:, 1::2] = torch.cos(pos * angle_rates)
        return pos_enc

def tokenize_text2text(texts, target_texts, tokenizer, max_length=50):
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
    
    input_ids = encodings['input_ids']
    decoder_input_ids = target_encodings['input_ids']
    return input_ids, decoder_input_ids

@pytest.mark.parametrize("num_samples", [100, 200])
def test_text_to_text(num_samples):
    dataset = load_real_data()

    subset = dataset.select(range(num_samples))  
    texts = [item['translation']['en'] for item in subset] 
    target_texts = [item['translation']['ro'] for item in subset]  
    input_ids, decoder_input_ids = tokenize_text2text(texts, target_texts, tokenizer)

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_ids = input_ids.cpu() #.numpy()
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(device)
    decoder_input_ids = decoder_input_ids.cpu() #.numpy()
    #labels = decoder_input_ids.clone()

    X_train_ids, X_test_ids, y_train, y_test = train_test_split(input_ids, decoder_input_ids, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train_ids, X_train_ids, y_train)
    test_dataset = TensorDataset(X_test_ids, X_test_ids, y_test)


    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    vocab_size = tokenizer.vocab_size
    model = TransformerSeq2SeqModel(vocab_size=vocab_size).to(device)

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
    test_text_to_text(100)