---
title: Transformers S12
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# Transformers_S12 🚀

**GPT-2 (124M) Overfitting Experiment**

This project demonstrates training a decoder-only transformer (GPT-2 Small) to **overfit** a dataset (`input.txt`). The goal is to achieve a training loss of **< 0.1**, effectively memorizing the input text.

## 📂 Contents

- **`main.py`**: Standalone Python script for training (CLI supported).
- **`train_overfit_colab_v1.ipynb`**: The main notebook. Contains the model definition (minGPT style), data loader, and training loop.
- **`input.txt`**: Training corpus (required).
- **`requirements.txt`**: Python dependencies.

## ⚡ Quick Start (Script)

To run the training script locally:

```bash
# Basic usage
python main.py --input_file input.txt

# Customize hyperparameters
python main.py --batch_size 16 --max_steps 5000 --lr 0.0005
```

## ⚡ Quick Start (Google Colab)

1.  Open `train_overfit_colab_v1.ipynb` in Google Colab.
2.  **Runtime** -> **Change runtime type** -> **T4 GPU**.
3.  Upload your `input.txt` file when prompted (or to the session storage).
4.  Run all cells.

The training loop is configured to stop automatically when the loss drops below `0.1`.

## 🛠️ Local Usage

To run locally, ensure you have a GPU environment set up.

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Launch Jupyter Lab/Notebook:
    ```bash
    jupyter lab train_overfit_colab_v1.ipynb
    ```
3.  Ensure `input.txt` is in the same directory.
4.  Run the notebook cells.

## 🧠 Model Details

- **Architecture**: GPT-2 Small (124M parameters)
- **Objective**: Causal Language Modeling (Next Token Prediction)
- **Strategy**: Overfitting (Train set = Validation set)