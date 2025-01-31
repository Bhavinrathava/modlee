import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts, get_device
from utils_tabular import *

device = get_device()
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

modlee_trainer_list = [False, True]

model_list = [TabularRegressionWithLayerNorm, TabularRegression, TabularRegressionWithDropout, TabularRegressionWideLeaky]
recommended_model_list = [True, False]

@pytest.mark.parametrize("load_data_func", [
    load_california_housing_data, 
    load_diabetes_data,           
    load_wine_quality_data       
])
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_type", model_list) 
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_tabular_regressor(load_data_func, modlee_trainer, model_type, recommended_model):
    X, y = load_data_func()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if recommended_model == True:

        recommender = modlee.recommender.from_modality_task(
            modality='tabular',
            task='regression'
            )
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = model_type(input_dim=X_train.shape[1]).to(device)

    if modlee_trainer == True:
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
    print(artifacts_path)

if __name__ == "__main__":
    test_tabular_regressor(load_california_housing_data, False, TabularRegressionWithLayerNorm, False)
