# SmolLM2-135M Reverse Engineering & Training

This project demonstrates how to reverse engineer the **SmolLM2-135M** model architecture, load official pretrained weights into a custom local implementation, and perform training with checkpointing and resumption.

## 🚀 Project Overview

The core objective is to understand the internals of the SmolLM2-135M model by:
1.  **Defining the Architecture**: Implementing a local `LlamaModel` class that matches the configuration of SmolLM2-135M (30 layers, 576 hidden dim, GQA).
2.  **Weight Transfer**: Mapping and loading weights from the official `HuggingFaceTB/SmolLM-135M` model into the local implementation.
3.  **Validation**: Verifying the local model produces identical (or highly similar) text generation results to the official HF model.
4.  **Training Workflow**:
    - Training on the `wikitext` dataset for 5000 steps.
    - Logging loss and text generation samples to TensorBoard.
    - Saving a checkpoint at step 5000.
    - Resuming training from the checkpoint for an additional 50 steps to verify state restoration.

## 🧠 Model Architecture Definition

The model is defined in `SmolLm3.py` and strictly follows the Llama architecture specifications optimized for the 135M parameter scale.

### Key Components:
- **Rotary Positional Embeddings (RoPE)**: 
  - Implemented in `RotaryPositionalEmbedding`.
  - Uses a `theta` of 10000.0 to encode positional information by rotating the query and key vectors.
  - Ensures the model can generalize to sequence lengths beyond what was seen during training (up to 2048/8192 context).

- **Grouped Query Attention (GQA)**:
  - Implemented in `LlamaAttention`.
  - **Heads**: 9 Query Heads, 3 Key/Value Heads.
  - This 3:1 ratio reduces memory bandwidth usage for the KV cache, speeding up inference while maintaining performance close to standard Multi-Head Attention.
  - The `forward` pass repeats the KV heads to match the number of Query heads before attention calculation.

- **SwiGLU Activation**:
  - Implemented in `LlamaMLP`.
  - Uses a Gated Linear Unit with the SiLU (Sigmoid Linear Unit) activation function.
  - Projects input into a higher dimension (Intermediate Size: 1536), applies the gate, and projects back to the hidden size (576).

- **RMSNorm**:
  - Implemented in `LlamaRMSNorm`.
  - Applies Root Mean Square Normalization before the attention and MLP blocks (Pre-Norm) for training stability.

## 📜 Training Logs & Verification

The training process is split into two phases to verify checkpointing and resumption capabilities.

### Phase 1: Initial Training (Steps 0 - 5000)
The model trains on the input dataset, logging loss and generating text samples.

**Sample Log Output:**
```text
Model loaded. Total params: 134515008
HF model params: 134515008
...
Step: 100, Train Loss: 11.5193
Step: 200, Train Loss: 11.7455
Step: 300, Train Loss: 10.7434
Step: 400, Train Loss: 10.8904
Step: 500, Train Loss: 10.7560

=== Generation @ step 500 ===
Once upon a time, they were less aware of a fairly challenging is very common activity, but some may befrequently altered objectives, has sni . is significant Skills tostairs that by other task creatively during whom, determined inflexu keyful, one critical analysethe attention movement argument relevant goal, in icty new or clearer and desirable encouragement and longer apetallehement influence of prosperity companionship or with analyse requirements
================================

Step: 600, Train Loss: 11.4689
Step: 700, Train Loss: 12.0275
Step: 800, Train Loss: 9.3529
Step: 900, Train Loss: 12.1106
Step: 1000, Train Loss: 10.7155

=== Generation @ step 1000 ===
Once upon a time, I had a few who like England II-S from Mexico Spearman, just had my personality spoke of that meaning. who somebody Cross Challenge tag-8'd swungn Have-a. One, original names property to Reasoning named appreciate if worry deeply. they been and things that,speaking the words what sailors to organise an oed what had suddenly and lean (Lateri, and he
================================

Step: 1100, Train Loss: 10.0201
Step: 1200, Train Loss: 11.7260
Step: 1300, Train Loss: 10.6011
Step: 1400, Train Loss: 11.8886
Step: 1500, Train Loss: 10.6372
...
WhichDr mantelt in food was had a
================================

Saved checkpoint_5000.pt
```

### Phase 2: Resumption (Steps 5001 - 5050)
The script loads `checkpoint_5000.pt` and continues training. 

**Verification of Continuity:**
The logs below demonstrate the model resuming exactly where it left off. The "Resume Loss" starts at a value consistent with the end of Phase 1, indicating the weights and state were correctly restored.

```text
Loaded checkpoint_5000.pt
Resume Training:  20%|█████████████▏                                                    | 10/50 [00:02<00:10,  3.96it/s]
Step: 10, Resume Loss: 12.4076
Resume Training:  40%|██████████████████████████▍                                       | 20/50 [00:06<00:09,  3.03it/s]
Step: 20, Resume Loss: 12.5138
Resume Training:  60%|███████████████████████████████████████▌                          | 30/50 [00:09<00:07,  2.76it/s]
Step: 30, Resume Loss: 10.7050
Resume Training:  80%|████████████████████████████████████████████████████▊             | 40/50 [00:11<00:02,  4.30it/s]
Step: 40, Resume Loss: 9.8365
Resume Training: 100%|██████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.51it/s]
Step: 50, Resume Loss: 11.5193
Saved checkpoint_5050.pt
```
*Note: The resume step counter resets to 0 in the logs, so "Step: 10" represents 10 steps **after** the checkpoint (Global Step 5010).*

## 📂 Files

- **`main.py`**: The main notebook (or script) containing the entire workflow: setup, loading, validation, training, and checkpointing.
- **`SmolLm3.py`**: Contains the custom `LlamaModel` class definition.
- **`SmolLm3_Config.yaml`**: Configuration file defining the model hyperparameters (layers, heads, dimensions, etc.).
- **`input.txt`**: The training data source.
