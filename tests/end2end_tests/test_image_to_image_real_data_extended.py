import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from utils import check_artifacts, get_device
from utils_image import load_dataset
from utils_image import AutoencoderDenoisingModel, UNetDenoisingModel, ResNetDenoisingModel, ModleeDenoisingModel

device = get_device()
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path='/home/ubuntu/efs/modlee_pypi_testruns')
model_classes = [AutoencoderDenoisingModel, AutoencoderDenoisingModel, UNetDenoisingModel, ResNetDenoisingModel, ModleeDenoisingModel]
recommended_model_list = [True,False]

@pytest.mark.parametrize("noise_level", [0.1])
@pytest.mark.parametrize("img_size", [(1, 32, 32),(3, 28, 28)]) 
@pytest.mark.parametrize("modlee_trainer", [False, True])
@pytest.mark.parametrize("dataset_name", ["CIFAR10", "MNIST", "FashionMNIST"])
@pytest.mark.parametrize("model_class", model_classes)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_image_to_image(noise_level, img_size, modlee_trainer, dataset_name, model_class, recommended_model):
    train_noisy_dataset, test_noisy_dataset = load_dataset(dataset_name, img_size, noise_level)
    train_dataloader = DataLoader(train_noisy_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_noisy_dataset, batch_size=32, shuffle=False)

    if recommended_model == True:
        recommender = modlee.recommender.from_modality_task(
            modality='image',
            task='imagetoimage', 
            img_size=img_size
            )

        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = model_class(img_size=img_size).to(device)

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
    test_image_to_image(0.1, (3, 28, 28), False, "CIFAR10", AutoencoderDenoisingModel, True)
