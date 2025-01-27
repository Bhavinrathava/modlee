import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils_timeseries import generate_dummy_time_series_data_forecasting
from utils_timeseries import MultivariateTimeSeriesForecaster, TransformerTimeSeriesForecaster
from utils import check_artifacts, get_device

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

parameter_combinations = [
    (5, 3, 2),
    (10, 6, 5),
    (10, 30, 10),
    (100, 20, 50)
]

modlee_trainer_list = [True, False]
model_classes = ['multivariate', 'transformer']
recommended_model_list = [True ,False]

@pytest.mark.parametrize("model_class", model_classes)
@pytest.mark.parametrize("num_features, seq_length, output_features", parameter_combinations)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_time_series_forecasting(model_class, num_features, seq_length, output_features, modlee_trainer, recommended_model):
    X, y = generate_dummy_time_series_data_forecasting(num_samples=1000, seq_length=seq_length, num_features=num_features, output_features=output_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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
        if model_class == 'multivariate':
            model = TransformerTimeSeriesForecaster(input_dim=num_features, seq_length=seq_length, output_dim=output_features, nhead=1).to(device)
        else:
            model = MultivariateTimeSeriesForecaster(input_dim=num_features, seq_length=seq_length, output_dim=output_features).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_time_series_forecasting(MultivariateTimeSeriesForecaster, 5, 10, 3, False)