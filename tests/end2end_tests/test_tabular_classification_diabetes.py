import os
import modlee
import lightning.pytorch as pl
import pytest
from utils import check_artifacts, get_device
from utils_tabular import get_diabetes_dataloaders
from utils_tabular import TabularClassifier

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

recommended_model_list = [True,False]
modlee_trainer_list = [True,False]

@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_tabular_classification(recommended_model,modlee_trainer):

    train_dataloader, val_dataloader, num_features, num_classes = get_diabetes_dataloaders()

    if recommended_model == True:
        recommender = modlee.recommender.TabularClassificationRecommender(num_classes=num_classes)
        recommender.fit(train_dataloader)
        modlee_model = recommender.model.to(device)        
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = TabularClassifier(input_dim=num_features, num_classes=num_classes).to(device)    

    if modlee_trainer == True:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_tabular_classification(recommended_model = True, modlee_trainer = True)