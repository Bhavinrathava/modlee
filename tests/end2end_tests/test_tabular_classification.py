import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts, get_device
from utils_tabular import generate_dummy_tabular_data_classification
from utils_tabular import TabularClassifier

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

num_samples_list = [100]
num_features_list = [20,100]
num_classes_list = [2,10]
recommended_model_list = [True,False]
modlee_trainer_list = [True,False]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("num_features", num_features_list)
@pytest.mark.parametrize("num_classes", num_classes_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_tabular_classification(num_samples, num_features, num_classes, recommended_model, modlee_trainer):
    X, y = generate_dummy_tabular_data_classification(num_samples=num_samples, num_features=num_features, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if recommended_model == True:
        recommender = modlee.recommender.TabularClassificationRecommender(num_classes=num_classes)
        recommender.fit(train_dataloader)
        modlee_model = recommender.model.to(device)        
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = TabularClassifier(input_dim=num_features, num_classes=num_classes).to(device)    

    if modlee_trainer == True:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=10)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )


    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_tabular_classification(100, 10, 2, True, False)