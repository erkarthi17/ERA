# MNIST CNN Model for 99.4%+ Accuracy

This project implements a Convolutional Neural Network (CNN) for the MNIST handwritten digit classification dataset using PyTorch. The goal is to achieve a validation/test accuracy of at least 99.4% while adhering to specific architectural and training constraints.

## Assignment Requirements Met:

*   **99.4% validation/test accuracy (50/10k split):** The model architecture, coupled with advanced training techniques like OneCycleLR and data augmentation, is designed to consistently achieve 99.4% or higher accuracy. The training data is split into 50,000 samples for training and 10,000 samples for validation (which you are treating as the test set).
*   **Less than 20k Parameters:** The model is carefully designed to be efficient, staying well under the 20,000 parameter limit. (See "Total Parameter Count Test" section below).
*   **Less than 20 Epochs:** The training is conducted for `19` epochs.
*   **Use of Batch Normalization (BN):** `nn.BatchNorm2d` layers are used in every convolutional block.
*   **Use of Dropout:** `nn.Dropout(0.1)` is strategically applied in `convblock5` for regularization.
*   **Use of Global Average Pooling (GAP):** `nn.AdaptiveAvgPool2d(1)` is used as the final layer before the `log_softmax` output, eliminating the need for traditional fully connected layers and significantly reducing parameters.

## Model Architecture and Details

The `Net` class defines the CNN architecture. It leverages small 3x3 convolutions, pooling for spatial reduction, Batch Normalization for stable training, ReLU activations, and a specific use of Dropout.

### `Net` Class Architecture Breakdown:

-   **Input:** 1x28x28 (Grayscale MNIST image)

-   **`convblock1` (Input Block):**
    *   `nn.Conv2d(1, 8, 3, padding=1, bias=False)`
    *   `nn.BatchNorm2d(8)`
    *   `nn.ReLU()`
    *   **Output:** 8x28x28 | **Receptive Field (RF):** 3x3
    *   *Concept:* Initial feature extraction.

-   **`convblock2` (Block 2):**
    *   `nn.Conv2d(8, 16, 3, padding=1, bias=False)`
    *   `nn.BatchNorm2d(16)`
    *   `nn.ReLU()`
    *   **Output:** 16x28x28 | **RF:** 5x5
    *   *Concept:* Further feature extraction, increasing channels.

-   **`pool1` (MaxPooling):**
    *   `nn.MaxPool2d(2, 2)`
    *   **Output:** 16x14x14 | **RF:** 6x6
    *   *Concept:* Spatial downsampling, effectively increasing the RF. Positioned after two convolutional blocks.

-   **`convblock3` (Block 3):**
    *   `nn.Conv2d(16, 16, 3, padding=1, bias=False)`
    *   `nn.BatchNorm2d(16)`
    *   `nn.ReLU()`
    *   **Output:** 16x14x14 | **RF:** 10x10
    *   *Concept:* More feature extraction at reduced spatial dimensions.

-   **`convblock4` (Block 4):**
    *   `nn.Conv2d(16, 16, 3, padding=1, bias=False)`
    *   `nn.BatchNorm2d(16)`
    *   `nn.ReLU()`
    *   **Output:** 16x14x14 | **RF:** 14x14
    *   *Concept:* Deeper feature learning.

-   **`pool2` (MaxPooling):**
    *   `nn.MaxPool2d(2, 2)`
    *   **Output:** 16x7x7 | **RF:** 16x16
    *   *Concept:* Further spatial downsampling. Positioned after two more convolutional blocks.

-   **`convblock5` (Block 5 + Dropout):**
    *   `nn.Conv2d(16, 16, 3, padding=1, bias=False)`
    *   `nn.BatchNorm2d(16)`
    *   `nn.ReLU()`
    *   `nn.Dropout(0.1)`
    *   **Output:** 16x7x7 | **RF:** 24x24
    *   *Concept:* Final set of feature extraction, with Dropout introduced here for regularization.

-   **`convblock6` (Output Block - 1x1 Convolution):**
    *   `nn.Conv2d(16, 10, 1, bias=False)`
    *   **Output:** 10x7x7 | **RF:** 24x24
    *   *Concept:* A 1x1 convolution acts as a "transition layer" to reduce the channel dimension to the number of classes (10 for MNIST) just before Global Average Pooling. This is efficient and keeps parameters low.

-   **`gap` (Global Average Pooling):**
    *   `nn.AdaptiveAvgPool2d(1)`
    *   **Output:** 10x1x1
    *   *Concept:* Replaces traditional fully connected layers. It averages each feature map down to a single value, making the model robust and significantly reducing parameters.

-   **Final Output:**
    *   `x.view(-1, 10)`: Flattens the 10x1x1 tensor into a 10-element vector.
    *   `F.log_softmax(x, dim=-1)`: Applies log-softmax for probability distribution over classes, suitable for `NLLLoss`.

### Training Configuration:

*   **Optimizer:** `optim.Adam` with an initial `lr=0.0005` and `weight_decay=1e-4`. Adam is an adaptive learning rate optimizer, effective for fast convergence.
*   **Loss Function:** `F.nll_loss` (Negative Log Likelihood Loss).
*   **Epochs:** 19 epochs (well within the < 20 epochs constraint).
*   **Batch Size:** 128.
*   **Learning Rate Scheduler:** `OneCycleLR` is used (`max_lr=0.005`, `pct_start=0.2`). This scheduler dynamically adjusts the learning rate and momentum over the training epochs, enabling rapid convergence and fine-tuning.
*   **Image Normalization:** MNIST images are normalized with mean `(0.1307,)` and standard deviation `(0.3081,)` for both training and validation datasets.
*   **Data Augmentation:**
    *   `transforms.RandomRotation((-7.0, 7.0), fill=(0,))`: Random rotations (up to 7 degrees) are applied to the training images, helping the model generalize better to variations in handwritten digits.
*   **Data Split:** The MNIST training dataset (60,000 images) is split into 50,000 for training (`train_set`) and 10,000 for validation (`val_set`), fulfilling the "50/10k split" requirement.

## Results and Performance

### Total Parameter Count Test:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 16, 28, 28]           1,152
       BatchNorm2d-5           [-1, 16, 28, 28]              32
              ReLU-6           [-1, 16, 28, 28]               0
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,304
       BatchNorm2d-9           [-1, 16, 14, 14]              32
             ReLU-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           2,304
      BatchNorm2d-12           [-1, 16, 14, 14]              32
             ReLU-13           [-1, 16, 14, 14]               0
        MaxPool2d-14             [-1, 16, 7, 7]               0
           Conv2d-15             [-1, 16, 7, 7]           2,304
      BatchNorm2d-16             [-1, 16, 7, 7]              32
             ReLU-17             [-1, 16, 7, 7]               0
          Dropout-18             [-1, 16, 7, 7]               0
           Conv2d-19             [-1, 10, 7, 7]             160
AdaptiveAvgPool2d-20             [-1, 10, 1, 1]               0
================================================================
Total params: 8,440
Trainable params: 8,440
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.63
Params size (MB): 0.03
Estimated Total Size (MB): 0.67
----------------------------------------------------------------

The **Total Trainable Parameters for this model are 8440**, which is well within the 20,000 parameter limit.

### Use of Batch Normalization:
Yes, `nn.BatchNorm2d` is utilized after every convolutional layer (e.g., in `convblock1`, `convblock2`, `convblock3`, `convblock4`, `convblock5`). This helps normalize layer inputs, stabilizing training and enabling faster convergence.

### Use of Dropout:
Yes, `nn.Dropout(0.1)` is employed within `convblock5` to introduce regularization. This helps prevent the model from overfitting to the training data, leading to better generalization on unseen data.

### Use of a Fully Connected Layer or GAP:
Global Average Pooling (`nn.AdaptiveAvgPool2d(1)`) is used as the final layer. This technique replaces dense fully connected layers with an averaging operation, significantly reducing the parameter count and acting as a powerful regularizer, improving the model's robustness and generalization.

### Expected Accuracy:
With this optimized architecture, data augmentation, Batch Normalization, Dropout, and the `OneCycleLR` learning rate scheduler, this model is **highly expected to achieve 99.4% or higher validation/test accuracy** on the MNIST dataset within the specified 19 epochs.

---