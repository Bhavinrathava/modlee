import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from utils import check_artifacts, get_device
from utils_image import generate_dummy_data_regression
from utils_image import ModleeImageRegression

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

num_samples_list = [100]
img_size_list = [(3, 32, 32), (1, 16, 16), (6, 16, 16)]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("img_size", img_size_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_image_regression(num_samples, img_size, modlee_trainer):
    
    X_train, y_train = generate_dummy_data_regression(num_samples=num_samples, img_size=img_size)
    X_test, y_test = generate_dummy_data_regression(num_samples=num_samples, img_size=img_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    modlee_model = ModleeImageRegression(img_size=img_size).to(device)

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
    test_image_regression(100, (3, 32, 32), False)
