import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import sys
import os
from tqdm import tqdm

from image_dataset import ImageDataset
from catornot_classifier import CatornotClassifier

def main():
    print('System Version:', sys.version)
    print('PyTorch version', torch.__version__)
    print('Torchvision version', torchvision.__version__)
    print('Numpy version', np.__version__)

    CWD = os.getcwd()
    IMG_TRAIN_PATH = os.path.join(CWD, 'img_dataset', 'train')
    IMG_TEST_PATH = os.path.join(CWD, 'img_dataset', 'test')

    if not os.path.isdir(IMG_TRAIN_PATH):
        raise FileNotFoundError(f'Training directory does not exist: {IMG_TRAIN_PATH}')
    if not os.path.isdir(IMG_TEST_PATH):
        raise FileNotFoundError(f'Testing directory does not exist: {IMG_TEST_PATH}')    
    
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
        
    train_dataset = ImageDataset(data_dir=IMG_TRAIN_PATH, transform=transform)
    test_dataset = ImageDataset(data_dir=IMG_TEST_PATH, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    image, label = train_dataset[100]
    print(image.shape)

    for images, labels in train_dataloader:
        break
    print(images.shape, " ", labels.shape)

    model = CatornotClassifier(2)
    example_out = model(images)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # run through dataset 5 times
    num_epochs = 5

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        ##### TRAINING #####
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_dataloader, desc='training'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # labels.size(0) gets batch size
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        ##### VALIDATION #####
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc='training'):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
    
        val_loss = running_loss / len(test_dataloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    torch.save(model.state_dict(), "catornot_model_weights.pth")


if __name__ == '__main__':
    main()