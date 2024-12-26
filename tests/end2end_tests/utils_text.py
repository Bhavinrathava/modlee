import modlee
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


'''
TEXT DATASETS
'''

def tokenize_texts(texts, tokenizer, max_length=20):
    encodings = tokenizer(
        texts,
        truncation=True,  
        padding="max_length",  
        max_length=max_length, 
        return_tensors="pt", 
        add_special_tokens=True,  
    )
    
    input_ids = encodings['input_ids'].to(torch.long) 
    attention_mask = encodings['attention_mask'].to(torch.long) 

    return input_ids, attention_mask

def generate_dummy_text_classification_data(num_samples=100, num_classes=2):
    example_texts = [
        "This is a positive example.",
        "This is a negative example.",
        "I enjoyed the experience.",
        "The product was disappointing.",
        "Outstanding service and support.",
        "Would not recommend this at all.",
        "Fantastic quality and design.",
        "Terrible quality, very disappointed.",
        "Absolutely loved the product!",
        "Not worth the price.",
    ]
    
    texts = [example_texts[i % len(example_texts)] for i in range(num_samples)]
    labels = torch.randint(0, num_classes, (num_samples,))  
    return texts, labels

def load_real_data(dataset_name="amazon_polarity"):
    if dataset_name == "amazon_polarity":
        dataset = load_dataset("amazon_polarity", split='train[:80%]')
        texts = dataset['content']
        targets = dataset['label'] 
        
    elif dataset_name == "yelp_polarity":
        dataset = load_dataset("yelp_polarity", split='train[:80%]')
        texts = dataset['text']
        targets = dataset['label']  
    
    return texts, targets

def generate_dummy_text_regression_data(num_samples=100):
    example_texts = [
        "The product met expectations.",
        "Quite satisfactory.",
        "An average experience overall.",
        "Not too bad, not too good.",
        "I liked it, but it could be better.",
        "Could be improved.",
        "This is acceptable.",
        "Not great, but decent.",
        "Meets basic needs.",
        "Could be more refined.",
    ]
    
    texts = [example_texts[i % len(example_texts)] for i in range(num_samples)]
    targets = torch.randn(num_samples)  
    return texts, targets

'''
TEXT MODELS
'''

class ModleeTextClassificationModel(modlee.model.TextClassificationModleeModel):
    def __init__(self, vocab_size, embed_dim=50, num_classes=2, tokenizer=None):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id if tokenizer else None)
        
        self.model = torch.nn.Sequential(
            self.embedding,  
            torch.nn.Flatten(),
            torch.nn.Linear(embed_dim * 20, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None):
        if isinstance(input_ids, list):
            input_ids = torch.cat(input_ids, dim=0)
        embedded = self.embedding(input_ids)
        for layer in list(self.model.children())[1:]:  
            embedded = layer(embedded)

        return embedded

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(preds, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(preds, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
class ModleeTextRegressionModel(modlee.model.TextRegressionModleeModel):
    def __init__(self, vocab_size, embed_dim=50, tokenizer=None):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id if tokenizer else None)
        
        self.model = torch.nn.Sequential(
            self.embedding,  
            torch.nn.Flatten(),
            torch.nn.Linear(embed_dim * 20, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)  
        )
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, input_ids, attention_mask=None):
        if isinstance(input_ids, list):
            input_ids = torch.cat(input_ids, dim=0)
        embedded = self.embedding(input_ids)
        for layer in list(self.model.children())[1:]:  
            embedded = layer(embedded)
        return embedded

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        preds = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(preds.squeeze(), targets)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        preds = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(preds.squeeze(), targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def initialize_model(model_type, vocab_size, tokenizer):
    """
    Initializes the model based on the model type.

    Args:
        model_type (str): Type of model ("transformer" or "automodel").
        vocab_size (int): Size of the vocabulary.
        tokenizer (AutoTokenizer): Tokenizer used for the model.

    Returns:
        nn.Module: The initialized model.
    """
    if model_type == "transformer":
        return TransformerSeq2SeqModel(vocab_size=vocab_size) #.to(device)
    elif model_type == "automodel":
        #return AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-tiny")  #sshleifer/tiny-mbart
        return ModleeText2TextAutoModel('sshleifer/tiny-mbart') #AutoModelForSeq2SeqLM.from_pretrained("sshleifer/tiny-mbart")  #

    else:
        raise ValueError(f"Invalid model type: {model_type}")

import torch.nn as nn
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
        return torch.optim.Adam(self.parameters(), lr=5e-5)

    @staticmethod
    def _generate_positional_encoding(d_model, max_length):
        pos = torch.arange(0, max_length).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle_rates = 1 / torch.pow(10000, (i.float() / d_model))
        pos_enc = torch.zeros(max_length, d_model)
        pos_enc[:, 0::2] = torch.sin(pos * angle_rates)
        pos_enc[:, 1::2] = torch.cos(pos * angle_rates)
        return pos_enc
    
class ModleeText2TextAutoModel(modlee.model.TextTextToTextModleeModel):
    def __init__(self, tokenizer, model_name="t5-small"):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        if isinstance(input_ids, list):
            #input_ids = torch.cat(input_ids, dim=0)
            input_ids, attention_mask, decoder_input_ids = input_ids
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        
        decoder_input_ids = self.model._shift_right(decoder_input_ids)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        
        logits = outputs.logits.cpu()
        return logits


    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self(input_ids, attention_mask, decoder_input_ids)
        
        # logits = logits
        # labels = labels 
        
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)).cpu(), labels.view(-1).cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, labels = batch
        logits = self(input_ids, attention_mask, decoder_input_ids)
        
        #logits = logits
        #labels = labels
        
        #logits = logits.to(torch.float32)
        #labels = labels.to(torch.long)  
        
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)).cpu(), labels.view(-1).cpu())
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
    