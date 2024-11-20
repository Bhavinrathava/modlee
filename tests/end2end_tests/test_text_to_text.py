import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pytest
from utils import check_artifacts
from utils import get_device

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path='/home/ubuntu/efs/modlee_pypi_testruns')

def generate_dummy_text2text_data(num_samples=100):
    example_texts = [
        "This is a positive example.",
        "I enjoyed the experience.",
        "Outstanding service and support.",
        "Fantastic quality and design.",
        "Absolutely loved the product!",
        "Would not recommend this at all.",
        "Terrible quality, very disappointed.",
        "Not worth the price."
    ]
    
    target_texts = [
        "C'est un exemple positif.",
        "J'ai apprécié l'expérience.",
        "Service et soutien exceptionnels.",
        "Qualité et conception fantastiques.",
        "J'ai absolument adoré le produit!",
        "Je ne recommanderais pas cela du tout.",
        "Qualité terrible, très déçu.",
        "Pas du tout rentable."
    ]
    
    texts = [example_texts[i % len(example_texts)] for i in range(num_samples)]
    target_texts = [target_texts[i % len(target_texts)] for i in range(num_samples)]
    
    return texts, target_texts

tokenizer = AutoTokenizer.from_pretrained("t5-small")

class ModleeText2TextModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, tokenizer, model_name="t5-small"):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cpu')
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        
        if isinstance(input_ids, list):
            input_ids = torch.cat(input_ids, dim=0)
        
        if decoder_input_ids is None:
            decoder_input_ids = input_ids 
        
        decoder_input_ids = self.model._shift_right(decoder_input_ids)
        input_ids = input_ids.to('cpu')
        type(input_ids)
        attention_mask = attention_mask.to('cpu')

        type(attention_mask)
        decoder_input_ids = decoder_input_ids.to('cpu')

        type(decoder_input_ids)
        self.model = self.model.to('cpu')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self.forward(input_ids, attention_mask, decoder_input_ids)
        
        logits = logits.to(torch.float32)
        labels = labels.to(torch.long) 
        
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self.forward(input_ids, attention_mask, decoder_input_ids)
        
        logits = logits.to(torch.float32)
        labels = labels.to(torch.long)  
        labels = labels.to(logits.device)
        
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
    texts, target_texts = generate_dummy_text2text_data(num_samples=num_samples)
    
    input_ids, attention_masks, decoder_input_ids = tokenize_text2text(texts, target_texts, tokenizer)
    target_ids = decoder_input_ids 
    
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

    modlee_model = ModleeText2TextModel(tokenizer=tokenizer).to('cpu')

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