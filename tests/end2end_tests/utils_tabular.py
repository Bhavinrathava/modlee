import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import random_split
'''

TABULAR DATASETS

'''

def generate_dummy_tabular_data_regression(num_samples=100, num_features=10):
    X = torch.randn(num_samples, num_features)
    y = torch.randn(num_samples) 
    return X, y

def generate_dummy_tabular_data_classification(num_samples=100, num_features=10, num_classes=2):
    """Generate dummy tabular data."""
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

def load_california_housing_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def load_diabetes_data():
    data = load_diabetes()
    X, y = data.data, data.target
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def load_wine_quality_data():
    data = fetch_openml(name="wine-quality-red", version=1)
    X = data.data.to_numpy()  
    y = data.target.astype(float).to_numpy()  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to tensors
        self.y = torch.tensor(y, dtype=torch.long) # Convert labels to long integers for classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_diabetes_dataloaders(batch_size=32, val_split=0.2, shuffle=True, num_classes=10):
    # Load the diabetes dataset from sklearn
    diabetes = load_diabetes()
    X = diabetes.data  # Features
    y = diabetes.target  # Continuous target (for regression)

    # Bin continuous target values into classes for classification
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), num_classes)) - 1  # Convert to class indices

    # Initialize the scaler for feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the features

    # Create a TabularDataset instance
    dataset = TabularDataset(X_scaled, y_binned)

    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, X.shape[1], num_classes  # Return num_features and num_classes

'''

TABULAR MODELS

'''

class TabularRegression(modlee.model.TabularRegressionModleeModel):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze() 
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class TabularRegressionWithDropout(modlee.model.TabularRegressionModleeModel):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze() 
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class TabularRegressionWithLayerNorm(modlee.model.TabularRegressionModleeModel):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LayerNorm(128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),   
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TabularRegressionWideLeaky(modlee.model.TabularRegressionModleeModel):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(128, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze() 
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TabularClassifier(modlee.model.TabularClassificationModleeModel):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


