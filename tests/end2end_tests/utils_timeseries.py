import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd


'''

TIME SERIES DATASETS

'''

def generate_dummy_time_series_data_regression(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, 1)  
    return X, y

def generate_dummy_time_series_data_classification(num_samples=1000, seq_length=20, num_features=10, num_classes=5):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

def generate_dummy_time_series_data_forecasting(num_samples=1000, seq_length=20, num_features=10, output_features=5):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, seq_length, output_features)
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

def load_ecg200_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values  
    X = data.iloc[:, 1:].values  
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def load_beef_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values 
    X = data.iloc[:, 1:].values 
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def load_car_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values  
    X = data.iloc[:, 1:].values  
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  
    y = torch.tensor(y, dtype=torch.long)
    return X, y

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
    
class MultivariateTimeSeriesClassifier(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(input_dim * seq_length, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(batch_size, -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
    
class SimpleTransformerModel(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, nhead=2, d_model=32):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.nhead = nhead

        if input_dim != d_model:
            self.input_proj = torch.nn.Linear(input_dim, d_model)
        else:
            self.input_proj = None
        self.d_model = d_model

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=2
        )
        self.fc = torch.nn.Linear(seq_length * d_model, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        if self.input_proj:
            x = self.input_proj(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))  
        x = x.permute(1, 0, 2).contiguous()  
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

class TransformerTimeSeriesClassifier(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, num_heads=1, hidden_dim=64):
        super().__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = torch.nn.Linear(input_dim * seq_length, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

class MultivariateTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, seq_length, -1)
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
    
class TransformerTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, nhead=2, num_layers=2):
        super().__init__()
        self.seq_length = seq_length
        self.transformer = torch.nn.Transformer(input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc_out = torch.nn.Linear(input_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = x.permute(1, 0, 2)  
        x = self.transformer(x, x)
        x = self.fc_out(x.permute(1, 0, 2)) 
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
    
class SequentialTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 30, hidden_dim),  # Flattened input sequence
            torch.nn.ReLU(),
            *[torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()) for _ in range(num_layers - 1)],
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten sequence
        return self.model(x).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# Model 4: Linear Forecaster
class LinearTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(seq_length * input_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MultivariateTimeSeriesForecasterV2(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(-1, input_dim)  # Flatten time dimension
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(batch_size, seq_length, -1)
        return x

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class TransformerTimeSeriesForecasterV2(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, nhead=4, num_layers=4, hidden_dim=128):
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_layers, dropout=0.2
        )
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.fc_in(x)  # Project input to hidden_dim
        x = x.permute(1, 0, 2)  # [seq_len, batch, feature]
        x = self.transformer(x, x)
        x = self.fc_out(x.permute(1, 0, 2))  # [batch, seq_len, output_dim]
        return x

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class SequentialTimeSeriesForecasterV2(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2):
        super().__init__()
        layers = [torch.nn.Linear(input_dim * 30, hidden_dim), torch.nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim // 2))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            hidden_dim = hidden_dim // 2  # Reduce dimensions gradually
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten sequence
        return self.model(x).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class LinearTimeSeriesForecasterV2(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(seq_length * input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
