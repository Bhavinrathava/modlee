import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
import pytest
from utils import check_artifacts, get_device
from utils_image import generate_dummy_segmentation_data
from utils_image import ImageSegmentation

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')
    
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("img_size, mask_size", [
    ((1, 32, 32), (32, 32)),
    ((3, 32, 32), (32, 32)),
    ((4, 16, 16), (16, 16)),
    ((5, 128, 128), (128, 128)),
])
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_image_segmentation(img_size, mask_size, modlee_trainer):
    X_train, y_train = generate_dummy_segmentation_data(num_samples=100, img_size=img_size, mask_size=mask_size)
    X_test, y_test = generate_dummy_segmentation_data(num_samples=20, img_size=img_size, mask_size=mask_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    in_channels = img_size[0] 
    modlee_model = ImageSegmentation(in_channels=in_channels).to(device)

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
    test_image_segmentation((3, 32, 32),(32, 32), False)