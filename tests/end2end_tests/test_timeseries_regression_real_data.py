import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import pytest
import pandas as pd
from utils import check_artifacts, get_device
from utils_timeseries import load_stock_data, load_power_consumption_data, load_shampoo_data
from utils_timeseries import TransformerTimeSeriesRegressor, MultivariateTimeSeriesRegressor
from sklearn.model_selection import train_test_split

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))
    
model_list = [TransformerTimeSeriesRegressor, MultivariateTimeSeriesRegressor]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("input_dim, seq_length, dataset_type", [
    (4, 3, 'stock'),
    (4, 6, 'stock'),
    (4, 30, 'stock'),
    (4, 20, 'stock'),
    (5, 3, 'power_consumption'),
    (5, 6, 'power_consumption'),
    (5, 30, 'power_consumption'),  
    (5, 20, 'power_consumption'),
    (1, 3, 'shampoo'),  
    (1, 6, 'shampoo'),
    (1, 30, 'shampoo'),
    (1, 20, 'shampoo')
])

@pytest.mark.parametrize("model_class", model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_time_series_regression(input_dim, seq_length, dataset_type, model_class, modlee_trainer):
    
    if dataset_type == 'stock':
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'A.csv')
        X, y = load_stock_data(file_path, seq_length)
    elif dataset_type == 'power_consumption':
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'powerconsumption.csv')
        X, y = load_power_consumption_data(file_path, seq_length)
    else:  
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'sales-of-shampoo-over-a-three-ye.csv')
        X, y = load_shampoo_data(file_path, seq_length)

    dataset = TensorDataset(X, y)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = model_class(input_dim=input_dim, seq_length=seq_length).to(device) 
    
    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader 
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_time_series_regression(4, 3, "stock", TransformerTimeSeriesRegressor, True)
