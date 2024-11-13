import os
import modlee
import pytest
import lightning.pytorch as pl
from utils import check_artifacts, get_device
from utils_image import image_classification_mnsit
from utils_image import ModleeImageClassification

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

img_size_list = [(3, 32, 32),(1, 64, 64)]
recommended_model_list = [True,False]
modlee_trainer_list = [True,False]

@pytest.mark.parametrize("img_size", img_size_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_image_classifer(img_size,recommended_model,modlee_trainer):
    train_dataloader, val_dataloader = image_classification_mnsit(img_size)
    num_classes = 10

    if recommended_model == True:
        recommender = modlee.recommender.ImageClassificationRecommender(num_classes=num_classes)
        recommender.fit(train_dataloader)
        modlee_model = recommender.model.to(device)
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = ModleeImageClassification(num_classes=num_classes, img_size=img_size).to(device)
    
    if modlee_trainer == True:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=5)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=2)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_image_classifer((3, 32, 32),True,True)