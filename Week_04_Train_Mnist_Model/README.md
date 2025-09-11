# 🧠 MNIST Digit Classifier (PyTorch)

This project trains a **Convolutional Neural Network (CNN)** on the MNIST handwritten digits dataset using **PyTorch**.  
It achieves **~98.5% accuracy on the test set in the very first epoch** 🚀.

---

## 📂 Project Structure
mnist_project/
│── data/ # MNIST dataset (auto-downloaded, ignored in git)
│── results/ # Plots & model summary saved here
│── main.py # Training & testing pipeline
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── .gitignore # Git ignore rules


---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist_project.git
   cd mnist_project

2. Install dependencies:

pip install -r requirements.txt

3. Usage

python main.py --epochs 10 --batch-size 256 --lr 0.0005

Available Options

--epochs : number of training epochs (default = 20)

--batch-size : batch size for training (default = 512)

--lr : learning rate (default = 0.001)

--step-size : StepLR scheduler step size (default = 15)

--gamma : StepLR gamma decay (default = 0.1)

--no-cuda : force CPU mode

4. Results

Test Accuracy: ~98.5% on Epoch 1

Accuracy and loss plots are saved in: results/training_curves.png

Model summary is saved in: results/model_summary.txt

5. .Net Class architecture as follows.,

Input: 1x28x28

Conv1: 1 → 32, kernel=3 → 26x26x32

Conv2: 32 → 64, kernel=3 → 24x24x64 → MaxPool(2x2) → 12x12x64

Conv3: 64 → 128, kernel=3 → 10x10x128

Conv4: 128 → 256, kernel=3 → 8x8x256 → MaxPool(2x2) → 4x4x256

Flatten: 4096

FC1: 4096 → 50

FC2: 50 → 10 (class logits)