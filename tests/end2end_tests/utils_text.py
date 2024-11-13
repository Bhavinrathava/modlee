import modlee
import torch
from datasets import load_dataset


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