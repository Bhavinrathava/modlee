import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import random
from torch.utils.data import Subset
from torchvision import datasets, transforms
'''

IMAGE DATASETS

'''
def generate_dummy_data_classification(num_samples=100, num_classes=2, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size, dtype=torch.float32)  
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long) 
    return X, y

def generate_dummy_data_regression(num_samples=100, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size, dtype=torch.float32)  
    y = torch.randn(num_samples, dtype=torch.float32) 
    return X, y

def generate_dummy_segmentation_data(num_samples=100, img_size=(3, 32, 32), mask_size=(32, 32)):
    X = torch.randn(num_samples, *img_size)
    y = torch.randint(0, 2, (num_samples, 1, *mask_size))
    return X, y

def generate_dummy_image_data_image_to_image(num_samples=100, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size, dtype=torch.float32)
    y = torch.randn(num_samples, *img_size, dtype=torch.float32)
    return X, y

def add_noise(img, noise_level=0.1):
    noise = torch.randn_like(img) * noise_level
    return torch.clamp(img + noise, 0., 1.)

class NoisyImageDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset, noise_level=0.1, img_size=(1, 32, 32)):
        self.dataset = dataset
        self.noise_level = noise_level
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        
        if img.size(0) != self.img_size[0]:
            if img.size(0) < self.img_size[0]:  
                img = img.repeat(self.img_size[0] // img.size(0), 1, 1) 
            else:  
                img = img[:self.img_size[0], :, :] 

        img = transforms.Resize((self.img_size[1], self.img_size[2]))(img)  
        noisy_img = add_noise(img, self.noise_level)
        return noisy_img, img  
    
def load_dataset(dataset_name, img_size, noise_level):
    transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[2])),
        transforms.ToTensor()
    ])

    train_size = 10000

    test_size=2000

    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_indices = random.sample(range(len(train_dataset)), train_size)
    test_indices = random.sample(range(len(test_dataset)), test_size)

    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_noisy_dataset = NoisyImageDataset(train_subset, noise_level=noise_level, img_size=img_size)
    test_noisy_dataset = NoisyImageDataset(test_subset, noise_level=noise_level, img_size=img_size)

    return train_noisy_dataset, test_noisy_dataset
    
AGE_GROUPS = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}

class AgeDataset(Dataset):
    def __init__(self, image_folder, csv_file, img_size, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]),  
            transforms.ToTensor(),
        ])
        if 'Class' in self.data.columns:
            self.data['Class'] = self.data['Class'].map(AGE_GROUPS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class PolygonDataset(Dataset):
    def __init__(self, image_folder, csv_file, img_size, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]), 
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx]['filename'])
        
        label = torch.tensor([
            self.data.iloc[idx]['bound_circle_x'], 
            self.data.iloc[idx]['bound_circle_y'], 
            self.data.iloc[idx]['bound_circle_r']
        ], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class BeautyRatingDataset(Dataset):
    def __init__(self, image_folder, txt_file, img_size, transform=None):
        self.image_folder = image_folder
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]),  
            transforms.ToTensor(),
        ])
        
        self.data = self.load_data(txt_file)

    def load_data(self, txt_file):
        data = []
        with open(txt_file, 'r') as file:
            for line in file:
                filename, rating = line.strip().split()
                data.append((filename, float(rating)))  
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data[idx][0])
        label = torch.tensor(self.data[idx][1], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
    
def image_classification_mnsit(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[2])),
        transforms.Grayscale(num_output_channels=img_size[0]),  # Convert images to RGB format
        transforms.ToTensor(),          # Convert images to tensors (PyTorch format)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std deviation
    ])

    train_dataset = datasets.MNIST( #this command gets the MNIST images
        root='./data',
        train=True, #loading the training split of the dataset
        download=True,
        transform=transform) #applies transformations defined earlier

    val_dataset = datasets.MNIST(
        root='./data',
        train=False, #loading the validation split of the dataset
        download=True,
        transform=transform)

    train_dataloader = DataLoader( #this tool loads the data
        train_dataset,
        batch_size=32, #we will load the images in groups of 4
        shuffle=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32)
    
    return train_dataloader, val_dataloader
'''

IMAGE MODELS

'''

class ModleeImageClassification(modlee.model.ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        input_dim = img_size[0] * img_size[1] * img_size[2]  
        
        self.model = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)  
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        
        if x.dim() == 5 and x.size(1) == 32:
            shape = x.shape
            x = x.view(-1, *shape) 
        x = self.model(x)
        return x
 
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        return {'loss':loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch  
        y_pred = self.forward(x) 
        val_loss = self.loss_fn(y_pred, y_target)  
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class ModleeImageRegression(modlee.model.ImageRegressionModleeModel):
    def __init__(self, img_size=(3, 32, 32)):
        super().__init__()
        self.img_size = img_size
        input_dim = img_size[0] * img_size[1] * img_size[2] 
        
        self.model = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(input_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, 1) 
        )
        
        self.loss_fn = nn.MSELoss()  

    def forward(self, x):
        if x.dim() == 5 and x.size(1) == 32:
            shape = x.shape
            x = x.view(-1, *shape)
        
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred, y_target)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
class ImageSegmentation(modlee.model.ImageSegmentationModleeModel):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output_size = x.shape[2:]  
        decoded = torch.nn.functional.interpolate(decoded, size=output_size, mode='bilinear', align_corners=False)
        return decoded

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ImageSegmentationV1(modlee.model.ImageSegmentationModleeModel):
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder: Add more layers and dropout
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),  # Dropout for regularization
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        # Decoder: Add symmetry to encoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output_size = x.shape[2:]  # Resize to match input dimensions
        decoded = torch.nn.functional.interpolate(decoded, size=output_size, mode='bilinear', align_corners=False)
        return decoded

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ImageSegmentationV2(modlee.model.ImageSegmentationModleeModel):
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder with wider layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        # Decoder with wider layers
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.nn.functional.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        return decoded

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ImageSegmentationV3(modlee.model.ImageSegmentationModleeModel):
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder with bottleneck
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        decoded = torch.nn.functional.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        return decoded

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ModleeImageToImageModel(modlee.model.ImageImagetoimageModleeModel):
    def __init__(self, img_size=(3, 32, 32)):
        super().__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Conv2d(img_size[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, img_size[0], kernel_size=3, stride=1, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred, y_target)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
class ModleeDenoisingModel(modlee.model.ImageImagetoimageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        self.img_size = img_size
        in_channels = img_size[0]  
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred, y_target)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class AutoencoderDenoisingModel(modlee.model.ImageImagetoimageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class UNetDenoisingModel(modlee.model.ImageImagetoimageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential( 
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, in_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x) 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class ResNetDenoisingModel(modlee.model.ImageImagetoimageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)