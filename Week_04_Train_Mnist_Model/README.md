# 🖤 MNIST Classification with PyTorch

This project trains a **Convolutional Neural Network (CNN)** on the classic [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/).  
It includes two architectures:  

- **`Net`** – A deep CNN ( > 1M parameters )  
- **`TinyNet`** – A lightweight CNN with **22,514 trainable parameters**  
  - Specially designed to stay **under 25k parameters** while still reaching **>95% accuracy in the first epoch** 🚀

---

## 📂 Project Structure

├── main.py # Training script
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── results/ # Saved plots & model summaries
│ ├── training_curves.png
│ └── model_summary.txt
└── data/ # MNIST dataset (auto-downloaded)

## ⚡ Installation

```bash
# Clone this repo
git clone [https://github.com/your-username/mnist-pytorch.git](https://github.com/erkarthi17/ERA.git)
cd mnist-pytorch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows 

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

Train with default settings:
----------------------------
   
   python main.py

Train with custom arguments:
----------------------------

   python main.py --epochs 5 --batch-size 256 --lr 0.01

Available arguments:
--------------------

--batch-size : Batch size (default = 512)

--epochs : Number of training epochs (default = 20)

--lr : Learning rate (default = 0.001)

--step-size : Scheduler step size (default = 15)

--gamma : Scheduler decay factor (default = 0.1)

--no-cuda : Force CPU training even if GPU is available

## 🏗️ TinyNet Architecture (<25k params)

The TinyNet architecture was carefully designed to balance efficiency and accuracy:

| Layer               | Output Shape      | Params |
| ------------------- | ----------------- | ------ |
| Conv2d (1→20) + BN  | 20 × 28 × 28      | 240    |
| MaxPool2d           | 20 × 14 × 14      | 0      |
| Conv2d (20→28) + BN | 28 × 14 × 14      | 5,104  |
| MaxPool2d           | 28 × 7 × 7        | 0      |
| Conv2d (28→64) + BN | 64 × 7 × 7        | 16,320 |
| AdaptiveAvgPool2d   | 64 × 1 × 1        | 0      |
| Fully Connected     | 10                | 650    |

| **Total Params**    | **22,514** ✅ <25k 

## 🔎 How it works

Conv1 + BN + Pooling – Extracts low-level features (edges, strokes).

Conv2 + BN + Pooling – Captures mid-level patterns (digit parts).

Conv3 + BN – Learns high-level digit representations.

Global Average Pooling (GAP) – Reduces feature maps to a compact vector.

FC Layer – Classifies into 10 digit classes (0–9).

📊 Results
-----------

   python main.py --epochs 5 --batch-size 256 --lr 0.01

   Test Accuracy after 1st epoch: ~95–96% ✅

   Final Accuracy (5 epochs): >98% 🎯

Training & evaluation curves are saved in results/training_curves.png.
