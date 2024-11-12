import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import pytest
from utils import check_artifacts, get_device
from utils_timeseries import load_ecg200_from_txt, load_beef_from_txt, load_car_from_txt
from utils_timeseries import MultivariateTimeSeriesClassifier, TransformerTimeSeriesClassifier

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

modlee_trainer_list = [True, False]
model_list = [MultivariateTimeSeriesClassifier, TransformerTimeSeriesClassifier]

@pytest.mark.parametrize("num_features, seq_length, num_classes, dataset_type", [
    (1,96,2,'ecg'),
    (1, 96, 6, 'ecg'),      
    (1, 470, 5,'beef'),
    (1, 470, 10, 'beef'),      
    (1, 577, 4, 'car'),
    (1, 577, 8, 'car'),
])
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_class", model_list)
def test_time_series_classification(num_features, seq_length, num_classes, dataset_type, modlee_trainer, model_class):
    
    if dataset_type == 'ecg':
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'ECG200_TRAIN.txt')
        X_train, y_train = load_ecg200_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'ECG200_TEST.txt')
        X_test, y_test = load_ecg200_from_txt(file_path)

    elif dataset_type == 'beef':
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Beef_TRAIN.txt')
        X_train, y_train = load_beef_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Beef_TEST.txt')
        X_test, y_test = load_beef_from_txt(file_path)

    else:
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Car_TRAIN.txt')
        X_train, y_train = load_car_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Car_TEST.txt')
        X_test, y_test = load_car_from_txt(file_path)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    modlee_model = model_class(input_dim=num_features, seq_length=seq_length, num_classes=num_classes).to(device)

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
