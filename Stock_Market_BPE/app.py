import gradio as gr
from tokenizer import StockBPE
import json

# Initialize and load the tokenizer
tokenizer = StockBPE()
try:
    tokenizer.load("stock_bpe")
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    # Fallback for initial build if files aren't there yet
    pass

def analyze_text(text):
    if not text:
        return "Please enter text", "0", "0.00x"
    
    # Encode
    tokens = tokenizer.encode(text)
    
    # Decode
    decoded = tokenizer.decode(tokens)
    
    # Stats
    original_len = len(text.encode('utf-8'))
    token_len = len(tokens)
    ratio = original_len / token_len if token_len > 0 else 0
    
    # Format output
    token_str = str(tokens)
    if len(token_str) > 1000:
        token_str = token_str[:1000] + "... (truncated)"
        
    return token_str, decoded, f"{ratio:.2f}x"

# Example data
examples = [
    ["TECH|AAPL|2020-11|MON|UNDER200|OPEN:113.9|HIGH:117.8|LOW:113.7|CLOSE:115.9|VOL:HIGH"],
    ["FIN|JPM|2023-05|FRI|UNDER150|OPEN:135.2|HIGH:136.5|LOW:134.8|CLOSE:135.9|VOL:MED"],
    ["TECH|MSFT|2024-01|WED|OVER300|OPEN:380.5|HIGH:385.2|LOW:379.0|CLOSE:384.8|VOL:HIGH"]
]

# Create Interface
iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter stock data here...", label="Input Text"),
    outputs=[
        gr.Textbox(label="Tokens IDs"),
        gr.Textbox(label="Decoded Back (Verification)"),
        gr.Label(label="Compression Ratio")
    ],
    title="📈 Stock Market BPE Tokenizer",
    description="A custom BPE tokenizer trained on financial time-series data. Enter stock data to see how it gets compressed!",
    examples=examples,
    theme="huggingface"
)

if __name__ == "__main__":
    iface.launch()
