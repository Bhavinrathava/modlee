import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from utils import check_artifacts, get_device
from utils_image import BeautyRatingDataset, PolygonDataset, AgeDataset
from utils_image import ModleeImageRegression

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

@pytest.mark.parametrize("img_size", [(3, 32, 32), (3, 64, 64), (3, 128, 128)])
@pytest.mark.parametrize("modlee_trainer", [True, False])
@pytest.mark.parametrize("dataset", ["age", "polygon", "beauty"])  
def test_image_regression(img_size, modlee_trainer, dataset):

    if dataset == "age":
        age_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'age')
        age_csv_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'age.csv')
        train_dataset = AgeDataset(age_image_folder, age_csv_file, img_size=img_size)
    elif dataset == "polygon":
        polygon_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'polygon')
        polygon_csv_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'polygon.csv')
        train_dataset = PolygonDataset(polygon_image_folder, polygon_csv_file, img_size=img_size)
    elif dataset == "beauty":
        beauty_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'facial_rating')
        beauty_txt_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'facial_rating.txt')
        train_dataset = BeautyRatingDataset(beauty_image_folder, beauty_txt_file, img_size=img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)

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
    test_image_regression(img_size=(3, 32, 32), modlee_trainer=True, dataset="age")

