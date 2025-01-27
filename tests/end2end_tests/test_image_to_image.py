import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from utils import check_artifacts, get_device
from utils_image import generate_dummy_image_data_image_to_image
from utils_image import ModleeImageToImageModel

device = get_device()
#modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path='/home/ubuntu/efs/modlee_pypi_testruns')

recommended_model_list = [True]

@pytest.mark.parametrize("num_samples", [100])
@pytest.mark.parametrize("img_size", [(1, 32, 32)])
                                      #(1, 28, 28), (3, 64, 64), (6, 128, 128)])
@pytest.mark.parametrize("modlee_trainer", [False])
@pytest.mark.parametrize("recommended_model", recommended_model_list)
def test_image_to_image(num_samples, img_size, modlee_trainer, recommended_model):
    print("img_size:", img_size)
    X_train, y_train = generate_dummy_image_data_image_to_image(num_samples=num_samples, img_size=img_size)
    X_test, y_test = generate_dummy_image_data_image_to_image(num_samples=num_samples, img_size=img_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
        modlee_model = ModleeImageToImageModel(img_size=img_size).to(device)

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
    test_image_to_image(100, (1, 32, 32), False, True)
