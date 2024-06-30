import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path

import requests

from timeit import default_timer as timer

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def show_image_from_loader(loader, batch_idx=0, idx=0, mean=None, std=None):
    """
    Function to display an image from a DataLoader batch.
    
    Args:
    - loader (DataLoader): PyTorch DataLoader object containing the dataset.
    - batch_idx (int): Index of the batch to visualize (default: 0).
    - idx (int): Index of the image within the batch to visualize (default: 0).
    - mean (list): Mean values used for normalization (default: None).
    - std (list): Standard deviation values used for normalization (default: None).
    
    Note:
    - If mean and std are provided, the function will perform unnormalization.
    """
    # Get the specified batch from DataLoader
    batch_idx, (X_batch, y_batch) = next(enumerate(loader))
    
    # Extract the specified image tensor from X_batch
    image = X_batch[idx]

    # If mean and std are provided, perform unnormalization
    if mean is not None and std is not None:
        unnormalize = transforms.Normalize(mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
                                           std=[1 / std[0], 1 / std[1], 1 / std[2]])
        image = unnormalize(image)

    # Convert tensor to numpy array
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # Reorder dimensions for matplotlib

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def download_file(url, save_as):
    """
    Function to download a file from a URL and save it locally.
    
    Args:
    - url (str): The URL of the file to download.
    - save_as (str): The file name/path to save the downloaded file.
    
    Returns:
    - None
    """
    # Check if the file already exists locally
    if Path(save_as).is_file():
        print(f"{save_as} already exists, skipping download")
    else:
        print(f"Downloading {save_as}")
        # Download the file
        request = requests.get(url)
        with open(save_as, "wb") as f:
            f.write(request.content)
        print(f"{save_as} downloaded successfully")
        
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}
    
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        
        if(batch % 10 == 0):
            print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples")

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time