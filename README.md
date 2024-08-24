
---

# README for Chest X-Ray Image Classification Notebook

## Introduction

This notebook demonstrates how to train and evaluate a Vision Transformer (ViT) and a ResNet50 model for chest X-ray image classification using PyTorch and TensorFlow. The dataset used is a chest X-ray dataset categorized into four classes: COVID-19, Normal, Pneumonia-Bacterial, and Pneumonia-Viral.

## Table of Contents

1. [Opening the Notebook](#opening-the-notebook)
2. [Dataset Overview](#dataset-overview)
3. [Setup and Imports](#setup-and-imports)
4. [Data Preparation](#data-preparation)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Saving the Model](#saving-the-model)
7. [CNN Model (ResNet50) Training](#cnn-model-resnet50-training)
8. [Conclusion](#conclusion)

## Opening the Notebook

1. **Accessing Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload this notebook to your Colab environment.

2. **Mount Google Drive**:
   - Ensure that your dataset is uploaded to your Google Drive.
   - Use the following code to mount your Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Dataset Overview

The dataset for this project is a chest X-ray image dataset categorized into four classes:

- **COVID-19**: X-ray images from patients diagnosed with COVID-19.
- **Normal**: X-ray images from healthy individuals.
- **Pneumonia-Bacterial**: X-ray images from patients with bacterial pneumonia.
- **Pneumonia-Viral**: X-ray images from patients with viral pneumonia.

### Download Links

- **Kaggle Dataset**: [Chest X-Ray Images Dataset](https://www.kaggle.com/datasets/jaberjaber/updatedxray/data)
- **Google Drive Link**: [Google Drive Folder](https://drive.google.com/drive/folders/1aWknbDU42nxixYnX0EA0z6ocINk3Wqgp?usp=sharing)

## Setup and Imports

Ensure the following libraries are installed:

- `torch`
- `torchvision`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `tqdm`
- `sklearn`
- `torchinfo`

Use the following code to import necessary libraries:

```python
import torch
from torch import nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm.auto import tqdm
import seaborn as sns
```

## Data Preparation

1. **Extract Dataset**:

   ```python
   import zipfile

   zip_ref = zipfile.ZipFile("/content/drive/MyDrive/archive (15).zip", 'r')
   zip_ref.extractall("tmp/")
   zip_ref.close()
   ```

2. **Set Directory Paths**:

   ```python
   data_paths = Path('/content/tmp')
   train_dir = data_paths / 'data_split/train'
   test_dir = data_paths / 'data_split/test'
   valid_dir = data_paths / 'data_split/validation'
   ```

3. **Create DataLoaders**:

   ```python
   def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):
       train_data = datasets.ImageFolder(train_dir, transform=transform)
       test_data = datasets.ImageFolder(test_dir, transform=transform)
       class_names = train_data.classes

       train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
       test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

       return train_dataloader, test_dataloader, class_names
   ```

4. **Define Transforms**:

   ```python
   transforms_train = transforms.Compose([
       transforms.Resize((IMG_SIZE, IMG_SIZE)),
       transforms.RandomHorizontalFlip(p=0.3),
       transforms.RandomResizedCrop(IMG_SIZE),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   ])

   transforms_test = transforms.Compose([
       transforms.Resize((IMG_SIZE, IMG_SIZE)),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   ])
   ```

5. **Visualize Sample Image**:

   ```python
   img, label = next(iter(train_dataloader))
   img, label = img[0], label[0]
   plt.imshow(img.permute(1,2,0))
   ```

## Model Training and Evaluation

1. **Define Helper Functions**:

   ```python
   def train_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
       # Training step implementation
       ...

   def test_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[Dict[str, float], List[int], List[int]]:
       # Testing step implementation
       ...

   def train(model: torch.nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int, device: torch.device) -> Dict[str, List]:
       # Training loop implementation
       ...

   def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
       # Save model implementation
       ...
   ```

2. **Train Vision Transformer (ViT) Model**:

   ```python
   import torchvision
   from torchinfo import summary

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   weights = torchvision.models.ViT_B_16_Weights.DEFAULT
   model = torchvision.models.vit_b_16(weights=weights).to(device)

   # Define loss function and optimizer
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

   results = train(model, train_dataloader, test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=EPOCHS, device=device)

   save_model(model=model, target_dir="models", model_name="vitdlmodel.pth")
   ```

3. **Train ResNet50 Model (using TensorFlow)**:

   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.keras.applications import ResNet50
   from tensorflow.keras.optimizers import RMSprop
   from tensorflow.keras.callbacks import EarlyStopping

   base_dir = '/content/tmp/data_split'
   train_dir = os.path.join(base_dir, 'train')
   valid_dir = os.path.join(base_dir, 'validation')
   test_dir = os.path.join(base_dir, 'test')

   train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
   validation_datagen = ImageDataGenerator(rescale=1./255)
   test_datagen = ImageDataGenerator(rescale=1./255)

   train_generator = train_datagen.flow_from_directory(train_dir, batch_size=32, class_mode="categorical", target_size=(224,224))
   validation_generator = validation_datagen.flow_from_directory(valid_dir, batch_size=32, class_mode="categorical", target_size=(224,224))
   test_generator = test_datagen.flow_from_directory(test_dir, batch_size=32, class_mode="categorical", target_size=(224,224))

   pre_trained_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
   ```

## Conclusion

This notebook demonstrates how to set up data loaders, train a Vision Transformer and a ResNet50 model for chest X-ray classification, and evaluate their performance. Make sure to adjust paths and parameters as needed for your specific setup.

---
