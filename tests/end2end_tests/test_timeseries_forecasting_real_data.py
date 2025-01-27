import os
import modlee
import torch
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from utils_timeseries import MultivariateTimeSeriesForecaster, TransformerTimeSeriesForecaster, SequentialTimeSeriesForecaster, LinearTimeSeriesForecaster
from utils_timeseries import MultivariateTimeSeriesForecasterV2, TransformerTimeSeriesForecasterV2, SequentialTimeSeriesForecasterV2, LinearTimeSeriesForecasterV2
import pytest
from utils import check_artifacts, get_device

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path='/home/ubuntu/efs/modlee_pypi_testruns')

modlee_trainer_list = [True, False]
recommended_model_list = [True ,False]
# Load and preprocess datasets
def load_real_data(dataset_name):
    urls = {
        "airline": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "shampoo": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
        "temps": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "sunspots": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
        "births": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv",
    }

    column_names = {"airline": "Passengers", "shampoo": "Sales","temps": "Temp", "sunspots": "Sunspots", "births": "Births"}

    url = urls[dataset_name]
    column = column_names[dataset_name]

    df = pd.read_csv(url)
    df[column] = df[column].astype(float)

    return df, column

def preprocess_time_series_data(df, column, seq_length, output_dim):
    values = df[column].values
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(normalized_values) - seq_length - output_dim):
        X.append(normalized_values[i:i + seq_length])
        y.append(normalized_values[i + seq_length:i + seq_length + output_dim])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

@pytest.mark.parametrize("model_class", ['multivariate', 'transformer', 'sequential', 'linear'])
@pytest.mark.parametrize("dataset_name", ['airline','shampoo', 'temps', 'sunspots', 'births'])
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_end2end_time_series_forecasting(model_class, dataset_name, modlee_trainer, recommended_model):
    df, column = load_real_data(dataset_name)
    seq_length, output_dim = 30, 1
    X, y = preprocess_time_series_data(df, column, seq_length, output_dim)
    train_size = int(0.8 * len(X))
    train_dataloader = DataLoader(TensorDataset(X[:train_size], y[:train_size]), batch_size=16)
    val_dataloader = DataLoader(TensorDataset(X[train_size:], y[train_size:]), batch_size=16)

    models = {
        'multivariate': MultivariateTimeSeriesForecaster(1, seq_length, output_dim),
        'transformer': TransformerTimeSeriesForecaster(input_dim=1, seq_length=seq_length, output_dim=output_dim, nhead=1),
        'sequential': SequentialTimeSeriesForecaster(input_dim=1, hidden_dim=64, output_dim=output_dim),
        'linear': LinearTimeSeriesForecaster(input_dim=1, seq_length=seq_length, output_dim=output_dim),
    }

    # models = {
    # 'multivariate': MultivariateTimeSeriesForecasterV2(1, seq_length, output_dim, hidden_dim=128),
    # 'transformer': TransformerTimeSeriesForecasterV2(input_dim=1, seq_length=seq_length, output_dim=output_dim, nhead=2, num_layers=4, hidden_dim=128),
    # 'sequential': SequentialTimeSeriesForecasterV2(input_dim=1, hidden_dim=128, output_dim=output_dim, num_layers=4, dropout=0.2),
    # 'linear': LinearTimeSeriesForecasterV2(input_dim=1, seq_length=seq_length, output_dim=output_dim, hidden_dim=128),
    # }

    if recommended_model == True:
        recommender = modlee.recommender.from_modality_task(
            modality='timeseries',
            task='forecasting', 
            prediction_length = 1
            )
        recommender.fit(train_dataloader)
        model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        model = models[model_class].to(device)
        
    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=40)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=40)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

    last_run_path = modlee.last_run_path()
    print(last_run_path)
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)
    

if __name__ == "__main__":
    test_end2end_time_series_forecasting('multivariate', 'temps', False, False)