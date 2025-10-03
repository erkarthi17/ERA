# --------------------------------------------------------------------------------------------------
# Text Block

# Purpose:
# This `train.py` file serves as the orchestrator for training the Convolutional Neural Network (CNN)
# model developed in `Model_CIFAR.py`. It centralizes the training logic,
# allowing for execution of model training and evaluation based on a command-line argument.

# Use Cases:
# - To run CIFAR_Model: `python train.py 1`
# - Each execution will train the specified model, print its summary, and plot its performance metrics.

# Structure:
# - Imports necessary libraries and `Net` class and `get_dataloaders` function from `Model_CIFAR.py` file.
# - Defines generic `train` and `test` functions to be used consistently across all models.
# - Contains a `run_model_training` function to encapsulate the model-specific setup (optimizer, scheduler)
#   and the training/evaluation loop.
# - Uses `argparse` to handle command-line arguments for selecting which model to train.
# --------------------------------------------------------------------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import OneCycleLR, StepLR
import argparse
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import Net class and data loader function from modular model file
from model_CIFAR import Net as CIFAR_Net, get_dataloaders as get_cifar_dataloaders


# Global lists to store results for plotting
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Generic Training Function
def train(model, device, train_loader, optimizer, epoch_num):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss.item()) # Store scalar
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100.*correct/processed:.2f}')
    train_acc.append(100.*correct/processed)


# Generic Testing Function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset):.2f}%)\n')
    test_acc.append(100. * correct / len(test_loader.dataset))

# Function to run training for any model
def run_model_training(model_name, ModelClass, get_dataloaders_func, epochs, optimizer_class, optimizer_params, scheduler_class=None, scheduler_params=None, initial_lr_info=None):
    global train_losses, test_losses, train_acc, test_acc
    train_losses, test_losses, train_acc, test_acc = [], [], [], [] # Reset for each model

    print(f"\n----- Training {model_name} -----")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    model = ModelClass().to(device)
    print(f"\nModel Summary for {model_name}:")
    summary(model, input_size=(3, 32, 32)) # Input size for CIFAR-10 is 3 channels, 32x32

    train_loader, test_loader = get_dataloaders_func(use_cuda)

    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = None
    if scheduler_class:
        if scheduler_class == OneCycleLR:
            # For OneCycleLR, steps_per_epoch is required and calculated from train_loader
            scheduler_params['steps_per_epoch'] = len(train_loader)
            scheduler = scheduler_class(optimizer, epochs=epochs, **scheduler_params)
        elif scheduler_class == StepLR:
            # CRITICAL FIX: Extract 'activate_after_epoch' before passing params to StepLR
            activate_after_epoch = scheduler_params.pop('activate_after_epoch', 0)
            scheduler = scheduler_class(optimizer, **scheduler_params)
            scheduler.activate_after_epoch = activate_after_epoch # Store it on the scheduler object
        # Add other scheduler types if needed

    print(f"Optimizer: {optimizer}")
    if scheduler:
        print(f"Scheduler: {scheduler}")
        if initial_lr_info:
            print(f"Initial LR for {model_name}: {initial_lr_info}")


    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}/{epochs}")
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        if scheduler:
            if scheduler_class == OneCycleLR:
                scheduler.step()
            elif scheduler_class == StepLR and (epoch + 1) >= scheduler.activate_after_epoch: # Use stored value
                scheduler.step()
                print(f"StepLR active. Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            elif scheduler_class == StepLR: # For epochs before activation
                print(f"StepLR not active. Current LR: {optimizer.param_groups[0]['lr']:.6f}")


    # Plotting results for the current model
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training and Test Metrics', fontsize=16)
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CIFAR model training.")
    parser.add_argument('model_id', type=int, help='Specify which model to run (1 for CIFAR_Model)')
    args = parser.parse_args()

    if args.model_id == 1: # Only CIFAR_Model is available
        run_model_training(
            model_name="CIFAR_Model",
            ModelClass=CIFAR_Net,
            get_dataloaders_func=get_cifar_dataloaders,
            epochs=20, # Increased epochs for better convergence
            optimizer_class=optim.AdamW, # Changed optimizer to AdamW
            optimizer_params={'lr': 0.001, 'weight_decay': 1e-2}, # Adjusted parameters for AdamW
            scheduler_class=OneCycleLR,
            scheduler_params={'max_lr': 0.01, 'pct_start': 0.3, 'div_factor': 10, 'final_div_factor': 100, 'anneal_strategy': 'cos'} # Tuned OneCycleLR parameters
        )
    else:
        print("Invalid model_id. Please choose 1 to run the CIFAR Model.")
        sys.exit(1)