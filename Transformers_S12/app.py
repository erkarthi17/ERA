import torch
import torch.nn.functional as F
import gradio as gr
import tiktoken
from main import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Setup & Model Loading
# -----------------------------------------------------------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f"Using device: {device}")

# Load model
config = GPTConfig()
model = GPT(config)

ckpt_path = 'ckpt_overfit.pth'
try:
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {ckpt_path}")
except FileNotFoundError:
    print(f"Checkpoint {ckpt_path} not found. Using random weights (or pretrained if you modify code).")
    # Optional: Load pretrained if no checkpoint
    # model = GPT.from_pretrained('gpt2') 

model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')

# -----------------------------------------------------------------------------
# Generation Function
# -----------------------------------------------------------------------------

def generate_text(prompt, max_new_tokens=50, temperature=0.8, top_k=200):
    if not prompt.strip():
        return "Please enter a prompt."
    
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)

    output_text = enc.decode(x[0].tolist())
    return output_text

# -----------------------------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demo = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(lines=2, label="Prompt", placeholder="Enter text here..."),
            gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Max New Tokens"),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
            gr.Slider(minimum=1, maximum=1000, value=200, step=1, label="Top-K"),
        ],
        outputs=gr.Textbox(label="Generated Text"),
        title="GPT-2 Overfitting Demo",
        description="This model is trained to overfit on a specific input text. Enter a prompt from that text to see it recite the memorized content.",
    )
    
    demo.launch()
