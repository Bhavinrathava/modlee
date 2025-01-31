import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
import pytest
from utils import check_artifacts, get_device
from utils_image import generate_dummy_segmentation_data
from utils_image import ImageSegmentation, ImageSegmentationV1, ImageSegmentationV2, ImageSegmentationV3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import tensorflow_datasets as tfds
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Subset
import random

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path='/home/ubuntu/efs/modlee_pypi_testruns')

# Models to test
models = [
    ImageSegmentation,
    ImageSegmentationV1,
    ImageSegmentationV2,
    ImageSegmentationV3
]

transform_image = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize masks
    transforms.ToTensor(),
])

# Datasets
def create_cityscapes_datasets():
    base_dir = "/home/ubuntu/efs/raw_data/archive"
    train_img_dir = os.path.join(base_dir, "train/img")
    train_label_dir = os.path.join(base_dir, "train/label")
    val_img_dir = os.path.join(base_dir, "val/img")
    val_label_dir = os.path.join(base_dir, "val/label")
    
    class CityscapesDataset(Dataset):
        def __init__(self, img_dir, label_dir, transform_image, transform_mask):
            self.img_dir = img_dir
            self.label_dir = label_dir
            self.transform_image = transform_image
            self.transform_mask = transform_mask
            self.images = sorted(os.listdir(self.img_dir))
            self.labels = sorted(os.listdir(self.label_dir))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.images[idx])
            label_path = os.path.join(self.label_dir, self.labels[idx])
            image = Image.open(img_path).convert("RGB")
            label = Image.open(label_path).convert("L")
            if self.transform_image:
                image = self.transform_image(image)
            if self.transform_mask:
                label = self.transform_mask(label)
            return image, label

    train_dataset = CityscapesDataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        transform_image=transform_image,
        transform_mask=transform_mask,
    )

    val_dataset = CityscapesDataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        transform_image=transform_image,
        transform_mask=transform_mask,
    )
    return train_dataset, val_dataset


def create_pascal_datasets():
    dataset = VOCSegmentation(
        root="./efs/raw_data",
        year="2012",
        image_set="train",
        download=True,
        transform=transform_image,
        target_transform=transform_mask,
    )
    return dataset, dataset  # Using same dataset for train and val for simplicity


def create_oxford_datasets():
    dataset = OxfordIIITPet(
        root="./efs/raw_data",
        split="trainval",
        target_types="segmentation",
        transform=transform_image,
        target_transform=transform_mask,
        download=True,
    )
    return dataset, dataset  # Using same dataset for train and val for simplicity


modlee_trainer_list = [False, True]
# Pytest parameterization
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_cls", models)
@pytest.mark.parametrize("dataset_creator", [create_cityscapes_datasets]) 
def test_image_segmentation(modlee_trainer, model_cls, dataset_creator):
    train_dataset, val_dataset = dataset_creator()

    # Create subsets for testing
    subset_size = int(len(train_dataset) * 0.1)  # Use 10% of the training data
    train_indices = random.sample(range(len(train_dataset)), subset_size)
    train_subset = Subset(train_dataset, train_indices)

    val_subset_size = int(len(val_dataset) * 0.1)  # Use 10% of the validation data
    val_indices = random.sample(range(len(val_dataset)), val_subset_size)
    val_subset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # Initialize the model
    in_channels = 3
    modlee_model = model_cls(in_channels=in_channels).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=30)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=30)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )


    # Check artifacts
    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, "artifacts")
    check_artifacts(artifacts_path)


if __name__ == "__main__":
    test_image_segmentation(False, ImageSegmentation, create_cityscapes_datasets)