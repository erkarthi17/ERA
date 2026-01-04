import gradio as gr
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from SmolLm3 import LlamaModel

# 1. Load Configuration
with open("SmolLm3_Config.yaml", "r") as f:
    full_cfg = yaml.safe_load(f)

model_cfg = full_cfg["model"]

# 2. Initialize Tokenizer
HF_MODEL = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Initialize Local Model
device = "cpu" # Spaces run on CPU by default unless GPU is selected
model = LlamaModel(model_cfg)

# 4. Load Pretrained Weights (Same logic as main.py)
print("Loading weights from Hugging Face...")
hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
hf_state = hf_model.state_dict()

def map_hf_to_local(hf_state):
    mapped = {}
    for k, v in hf_state.items():
        if k.startswith("model."):
            new_k = k.replace("model.", "")
        else:
            new_k = k
        mapped[new_k] = v
    return mapped

mapped_state = map_hf_to_local(hf_state)
model.load_state_dict(mapped_state, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully.")

# 5. Define Generation Function
def generate_text(prompt, max_new_tokens, temperature, top_k):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Use the model's generate method
    # Note: The custom model's generate method signature in SmolLm3.py is:
    # generate(self, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None)
    
    context_length = model_cfg["model_config"]["max_position_embeddings"]
    
    generated_ids = model.generate(
        idx=input_ids,
        max_new_tokens=int(max_new_tokens),
        context_length=context_length,
        temperature=float(temperature),
        top_k=int(top_k) if top_k > 0 else None,
        eos_token=tokenizer.eos_token_id,
        device=device
    )
    
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text

# 6. Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM2-135M Custom Implementation Demo")
    gr.Markdown("This demo uses a custom `LlamaModel` implementation running with weights loaded from `HuggingFaceTB/SmolLM-135M`.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", value="Once upon a time,")
            max_tokens_slider = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Max New Tokens")
            temp_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            top_k_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-K (0 to disable)")
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            output_text = gr.Textbox(label="Generated Output", lines=10)
            
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_slider, temp_slider, top_k_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
