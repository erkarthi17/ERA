import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

def get_data_loaders(
    batch_size,
    num_workers,
    train_subset=None,
    val_subset=None,
):
    """
    Returns data loaders for training and validation using a LOCAL copy of 'imagenette'.
    This bypasses the Hugging Face 'datasets' library to avoid download/cache issues.
    """
    # --- This now points to a local directory ---
    local_data_path = './imagenette2'
    train_path = os.path.join(local_data_path, 'train')
    val_path = os.path.join(local_data_path, 'val')

    # --- Check if the local data exists ---
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("\n" + "="*80)
        print(f"FATAL ERROR: Local dataset not found at '{local_data_path}'")
        print("This script now uses a local dataset to avoid Hugging Face download issues.")
        print("\nPLEASE FOLLOW THESE STEPS:")
        print("1. Download the dataset from this URL:")
        print("   https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
        print("\n2. Extract the downloaded 'imagenette2.tgz' file using a tool like 7-Zip or WinRAR.")
        print("\n3. Place the extracted 'imagenette2' folder inside your project directory:")
        print(f"   {os.getcwd()}")
        print("\n4. Ensure the following paths exist:")
        print(f"   - {train_path}")
        print(f"   - {val_path}")
        print("="*80)
        raise FileNotFoundError(f"Dataset not found. Please download and extract it to '{local_data_path}'.")

    print(f"Loading dataset from local path: '{local_data_path}'")

    # Define standard ImageNet transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Use torchvision.datasets.ImageFolder ---
    # This is a robust way to load image data from a local directory.
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    # --- Create Data Loaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print("Data loaders created successfully from local data.")
    return train_loader, val_loader