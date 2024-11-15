import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset  
import pytest
from utils import check_artifacts
from utils import get_device


device = get_device()
#modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path='/home/ubuntu/efs/modlee_pypi_testruns')
modlee.init(api_key='MsvVailphdUF2pcwbRRqrhx7ibxGC05W', run_path='/home/ubuntu/efs/modlee_pypi_testruns')

def load_real_data():
    dataset = load_dataset("wmt16", "ro-en", split='train[:80%]')
    print(f"Dataset structure: {dataset}")
    return dataset

tokenizer = AutoTokenizer.from_pretrained("t5-small")

class ModleeText2TextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, tokenizer, model_name="t5-small"):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        if isinstance(input_ids, list):
            input_ids = torch.cat(input_ids, dim=0)
        
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        
        decoder_input_ids = self.model._shift_right(decoder_input_ids)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        
        logits = outputs.logits
        return logits


    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self.forward(input_ids, attention_mask, decoder_input_ids)
        
        logits = logits
        labels = labels 
        
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self.forward(input_ids, attention_mask, decoder_input_ids)
        
        logits = logits
        labels = labels
        
        #logits = logits.to(torch.float32)
        #labels = labels.to(torch.long)  
        
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)


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
    
    input_ids = encodings['input_ids'].to(torch.long)
    attention_mask = encodings['attention_mask'].to(torch.long)
    decoder_input_ids = target_encodings['input_ids'].to(torch.long)
    
    decoder_input_ids = torch.cat([
        decoder_input_ids[:, :1],
        decoder_input_ids[:, 1:]
    ], dim=1)

    return input_ids, attention_mask, decoder_input_ids

@pytest.mark.parametrize("num_samples", [100, 200])
def test_text_to_text(num_samples):
    dataset = load_real_data()

    subset = dataset.select(range(num_samples))  
    texts = [item['translation']['en'] for item in subset] 
    target_texts = [item['translation']['ro'] for item in subset]  
    input_ids, attention_masks, decoder_input_ids = tokenize_text2text(texts, target_texts, tokenizer)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(device)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(device)

    target_ids = decoder_input_ids 
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)

    X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
        input_ids, attention_masks, target_ids, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_ids, dtype=torch.long).to(device).cpu(),
        torch.tensor(X_train_masks, dtype=torch.long).to(device).cpu(),
        torch.tensor(X_train_ids, dtype=torch.long).to(device).cpu(),
        torch.tensor(y_train, dtype=torch.long).to(device).cpu()
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_ids, dtype=torch.long).to(device).cpu(),
        torch.tensor(X_test_masks, dtype=torch.long).to(device).cpu(),
        torch.tensor(X_test_ids, dtype=torch.long).to(device).cpu(),
        torch.tensor(y_test, dtype=torch.long).to(device).cpu()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    modlee_model = ModleeText2TextModel(tokenizer=tokenizer).to(device)
    modlee_model = modlee_model

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
    test_text_to_text(100)