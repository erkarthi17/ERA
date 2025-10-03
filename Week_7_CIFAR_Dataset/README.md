# Week_7_CIFAR_Dataset: Advanced CNN Architecture for CIFAR-10 Classification

## Project Overview

This project focuses on developing an advanced Convolutional Neural Network (CNN) architecture for the CIFAR-10 image classification dataset. The primary goal is to achieve high accuracy (over 85%) while adhering to strict architectural constraints, including a limited parameter count (under 200K), specific convolutional block designs (no max pooling, use of 1x1 convolutions, dilated convolutions, and depthwise separable convolutions), and robust data augmentation using the Albumentations library.

The project demonstrates an efficient approach to designing deep learning models that balance performance with computational efficiency, crucial for deployment in resource-constrained environments.

## Features

The model architecture and training strategy incorporate the following key features:

*   **Custom CNN Blocks**: Designed with a series of convolutional layers, utilizing strided convolutions for downsampling instead of traditional max-pooling, allowing for greater control over feature extraction.
*   **1x1 Convolutions**: Strategically integrated for channel reduction and increased model depth without significantly increasing computational cost.
*   **Dilated Convolutions**: One or more layers incorporate dilated convolutions to expand the receptive field effectively, capturing broader context without increasing the number of parameters or losing resolution.
*   **Depthwise Separable Convolutions**: At least one layer utilizes depthwise separable convolutions to significantly reduce the parameter count and computational complexity while maintaining representational power.
*   **Global Average Pooling (GAP)**: Compulsory use of GAP to reduce spatial dimensions before the final classification layer, acting as a strong regularizer and reducing parameters.
*   **Fully Connected Layer**: A 1x1 convolution after GAP serves as the final fully connected layer, mapping features to 10 output classes for CIFAR-10.
*   **Albumentations Data Augmentation**: Comprehensive data augmentation pipeline to improve model generalization and robustness:
    *   `HorizontalFlip`: Randomly flips images horizontally.
    *   `ShiftScaleRotate`: Applies random shifts, scaling, and rotations.
    *   `CoarseDropout`: Randomly masks out square regions of the image with specific parameters (`max_holes=1`, `max_height=16`, `max_width=16`, `min_holes=1`, `min_height=16`, `min_width=16`, `fill_value=mean of dataset`).
    *   Normalization with CIFAR-10 mean and standard deviation.
*   **Parameter Efficiency**: The entire model is constrained to have **less than 200,000 trainable parameters**.
*   **High Accuracy Target**: Achieves **over 85% test accuracy**.
*   **Sufficient Receptive Field**: The total receptive field (RF) of the network is calculated to be **greater than 44**.

## Project Structure

Week_7_CIFAR_Dataset/
├── model_CIFAR.py # Defines the CNN architecture and data loading/augmentation logic
├── train.py # Main script to orchestrate training, testing, and plotting
└── requirements.txt # Lists all Python dependencies


## Setup

To set up and run this project, follow these steps:

1.  **Navigate to the Project Directory**:
    ```bash
    cd Week_7_CIFAR_Dataset
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Install all required Python packages using `pip` and the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` includes `numpy<2` to address potential compatibility issues with `torch` and `matplotlib` if you have a newer NumPy version.*

## Usage

To train the CIFAR-10 model, run the `train.py` script with `model_id` set to `1`:

```bash
python train.py 1
```

The script will:
*   Initialize the `CIFAR_Net` model.
*   Print a summary of the model architecture, including layer-wise output shapes and parameter counts (`torchsummary` output).
*   Download the CIFAR-10 dataset (if not already present).
*   Train the model for a specified number of epochs (default is 100 in `train.py`).
*   Evaluate the model's performance on the test set after each epoch.
*   Plot the training/test loss and accuracy curves upon completion.

## Model Performance

Based on the implemented architecture and training strategy:

*   **Total Parameters**: **186,784** (well within the <200K limit)
*   **Total Receptive Field**: **57** (greater than the 44 requirement)
*   **Achieved Test Accuracy**: **>85.4%** (well within 20 Epochs)

These results confirm that the model successfully meets all the specified requirements, demonstrating an effective and efficient approach to CIFAR-10 classification.