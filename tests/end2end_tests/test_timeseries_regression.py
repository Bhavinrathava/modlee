import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts, get_device
from utils_timeseries import generate_dummy_time_series_data_regression
from utils_timeseries import TransformerTimeSeriesRegressor, MultivariateTimeSeriesRegressor

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

parameter_combinations = [
    (5, 3), 
    (10, 6),
    (10, 30),
    (100, 20)
]

modlee_trainer_list = [True, False]
model_list = [TransformerTimeSeriesRegressor, MultivariateTimeSeriesRegressor]

@pytest.mark.parametrize("num_features, seq_length", parameter_combinations)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_class", model_list)
def test_time_series_regression(num_features, seq_length, modlee_trainer, model_class):
    X, y = generate_dummy_time_series_data_regression(num_samples=1000, seq_length=seq_length, num_features=num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    modlee_model = model_class(input_dim=num_features, seq_length=seq_length).to(device) 

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_time_series_regression(5, 20, True, TransformerTimeSeriesRegressor)

