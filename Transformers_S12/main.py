import os
import math
import time
import inspect
from dataclasses import dataclass
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        # spelling must match the check in _init_weights (NANGPT_SCALE_INIT)
        self.c_proj.NANGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, input_file='input.txt'):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        # OVERFIT STRATEGY: Use ALL tokens for training AND validation
        self.train_tokens = self.tokens
        self.val_tokens = self.tokens

        print(f'loaded {len(self.tokens)} tokens (train={len(self.train_tokens)}, val={len(self.val_tokens)})')
        print(f'1 epoch = {len(self.train_tokens) // (B * T)} batches')

        # state
        self.current_position = 0
        self.val_position = 0

    def next_batch(self, split='train'):
        B, T = self.B, self.T
        # Always use tokens (which are same for train/val now)
        tokens = self.tokens

        if split == 'train':
            pos = self.current_position
        else:
            pos = self.val_position

        if pos + (B * T + 1) > len(tokens):
            pos = 0

        buf = tokens[pos: pos + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        new_pos = pos + B * T
        if new_pos + (B * T + 1) > len(tokens):
            new_pos = 0

        if split == 'train':
            self.current_position = new_pos
        else:
            self.val_position = new_pos

        return x, y

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def evaluate(model, data_loader, device, use_amp=False, max_batches=None):
    model.eval()
    losses = []
    with torch.no_grad():
        num = 0
        while True:
            if max_batches is not None and num >= max_batches:
                break
            try:
                x, y = data_loader.next_batch(split='val')
            except Exception:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, loss = model(x, y)
            losses.append(loss.item())
            num += 1
            # stop if we've looped over the validation tokens (which are now all tokens)
            if data_loader.val_position == 0:
                break
    model.train()
    return float(sum(losses) / max(1, len(losses))) if losses else float('inf')

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 to overfit on a text file.")
    parser.add_argument('--input_file', type=str, default='input.txt', help='Path to input text file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--max_steps', type=int, default=20000, help='Maximum training steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--val_interval', type=int, default=500, help='Validation interval')
    parser.add_argument('--target_loss', type=float, default=0.099999, help='Target validation loss to stop training')
    parser.add_argument('--ckpt_path', type=str, default='ckpt_overfit.pth', help='Checkpoint path')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained GPT-2 weights (not implemented in this script, uses scratch)')

    args = parser.parse_args()

    # Initialize
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        device = 'cuda'
        print("Using CUDA")
    else:
        device = 'cpu'
        print("Using CPU (Warning: Training will be very slow)")

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    data = DataLoaderLite(B=args.batch_size, T=args.seq_len, input_file=args.input_file)
    
    # Note: The notebook had a from_pretrained option but the main flow used scratch.
    # We'll default to scratch as per the notebook's main execution flow, 
    # but could add logic for pretrained if needed.
    model = GPT(GPTConfig()) 
    model.to(device)

    print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

    # Training
    pbar = tqdm(total=args.max_steps, desc='training')
    step = 0
    best_val = float('inf')

    while step < args.max_steps:
        # gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(args.accum_steps):
            x, y = data.next_batch(split='train')
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()
        step += 1
        pbar.update(1)
        if step % args.log_interval == 0:
            pbar.set_postfix({'train_loss': f"{total_loss:.6f}", 'step': step})

        if step % args.val_interval == 0:
            val_loss = evaluate(model, data, device, use_amp, max_batches=32)
            print(f"\n[val] step {step} loss {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.ckpt_path)
                print(f"Saved best checkpoint (val {best_val:.6f}) to {args.ckpt_path}")
            if val_loss < args.target_loss:
                print(f"Target val loss {args.target_loss} reached at step {step} (val {val_loss:.6f}).")
                break

    pbar.close()
    print(f"Training finished. Final step {step}, best val {best_val}")

if __name__ == "__main__":
    main()
