import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd


'''

TIME SERIES DATASETS

'''

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, 1)  
    return X, y


def load_shampoo_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Month'] = pd.to_datetime(data['Month'], format='%d-%b')  
    data.set_index('Month', inplace=True)

    y = data['Sales of shampoo over a three year period'].values  
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = len(y) - seq_length + 1
    y_seq = torch.stack([y[i:i + seq_length] for i in range(num_samples)])

    X_seq = torch.zeros(num_samples, seq_length, 1) 

    return X_seq, y_seq

def load_stock_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    X = data[['Open', 'High', 'Low', 'Volume']].values  
    y = data['Close'].values  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = X.shape[0] - seq_length + 1
    X_seq = torch.stack([X[i:i + seq_length] for i in range(num_samples)])
    y_seq = y[seq_length - 1:]  

    return X_seq, y_seq

def load_power_consumption_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    
    X = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values
    y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1).values  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = X.shape[0] - seq_length + 1
    X_seq = torch.stack([X[i:i + seq_length] for i in range(num_samples)])
    y_seq = y[seq_length - 1:]  

    return X_seq, y_seq



'''

TIME SERIES MODELS

'''



class MultivariateTimeSeriesRegressor(modlee.model.TimeseriesRegressionModleeModel):
    def __init__(self, input_dim, seq_length, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(input_dim * seq_length, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(batch_size, -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.mse_loss(preds, y) 
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class TransformerTimeSeriesRegressor(modlee.model.TimeseriesRegressionModleeModel):
    def __init__(self, input_dim, seq_length, num_heads=1, hidden_dim=64):
        super().__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = torch.nn.Linear(input_dim * seq_length, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)