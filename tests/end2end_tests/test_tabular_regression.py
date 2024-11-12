import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts, get_device
from utils_tabular import generate_dummy_tabular_data_regression
from utils_tabular import TabularRegression

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

num_samples_list = [100, 200, 300]
num_features_list = [10, 20, 50, 100]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("num_features", num_features_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_tabular_regression(num_samples, num_features, modlee_trainer):    
    X, y = generate_dummy_tabular_data_regression(num_samples=num_samples, num_features=num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    modlee_model = TabularRegression(input_dim=num_features).to(device)

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
    test_tabular_regression(100, 10, False)
