# Solving for residual std scaling issue in minGPT implementation
import os
import math
import time
import inspect
from dataclasses import dataclass
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from transformers import GPT2LMHeadModel
from tqdm.auto import trange, tqdm

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

# model = GPT.from_pretrained('gpt2')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # create train / validation split (default 95/5)
        self.val_split = int(0.05 * len(self.tokens))
        if self.val_split < (B * T + 1):
            # if dataset too small for a val split, set val_split to zero
            self.val_split = 0
        if self.val_split > 0:
            self.val_tokens = self.tokens[-self.val_split:]
            self.train_tokens = self.tokens[:-self.val_split]
        else:
            self.val_tokens = torch.tensor([], dtype=self.tokens.dtype)
            self.train_tokens = self.tokens

        print(f'loaded {len(self.tokens)} tokens (train={len(self.train_tokens)}, val={len(self.val_tokens)})')
        print(f'1 epoch = {len(self.train_tokens) // (B * T)} batches')

        # state
        self.current_position = 0
        self.val_position = 0
    
    def next_batch(self, split='train'):
        B, T = self.B, self.T
        if split == 'train' or len(self.val_tokens) == 0:
            tokens = self.train_tokens
            pos = self.current_position
            if pos + (B * T + 1) > len(tokens):
                pos = 0
            buf = tokens[pos: pos + B * T + 1]
            x = (buf[:-1]).view(B, T)
            y = (buf[1:]).view(B, T)
            self.current_position = pos + B * T
            if self.current_position + (B * T + 1) > len(tokens):
                self.current_position = 0
            return x, y
        else:
            # validation: deterministic sequential batches
            tokens = self.val_tokens
            pos = self.val_position
            if pos + (B * T + 1) > len(tokens):
                pos = 0
            buf = tokens[pos: pos + B * T + 1]
            x = (buf[:-1]).view(B, T)
            y = (buf[1:]).view(B, T)
            self.val_position = pos + B * T
            if self.val_position + (B * T + 1) > len(tokens):
                self.val_position = 0
            return x, y


model = GPT(GPTConfig())
model.to(device)

train_loader = DataLoaderLite(B = 4, T = 32)
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
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, y)
            losses.append(loss.item())
            num += 1
            # stop if we've looped over the validation tokens
            if data_loader.val_position == 0:
                break
    model.train()
    return float(sum(losses) / max(1, len(losses))) if losses else float('inf')


def save_checkpoint(path, model, optimizer, scaler, step, best_val):
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'step': step,
        'best_val': best_val,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt and ckpt['optimizer_state'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler is not None and ckpt.get('scaler_state') is not None:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt.get('step', 0), ckpt.get('best_val', float('inf'))


def train(args):
    # re-create data loader with user B, T
    data = DataLoaderLite(B=args.batch_size, T=args.seq_len)

    # model (optionally pretrained)
    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained_model}")
        model_local = GPT.from_pretrained(args.pretrained_model)
    else:
        model_local = GPT(GPTConfig())
    model_local.to(device)

    optimizer = torch.optim.AdamW(model_local.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    best_val = float('inf')
    if args.resume and os.path.exists(args.ckpt_path):
        print(f"Loading checkpoint {args.ckpt_path}")
        start_step, best_val = load_checkpoint(args.ckpt_path, model_local, optimizer, scaler)

    pbar = tqdm(total=args.max_steps, desc='training')
    step = start_step
    while step < args.max_steps:
        # gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(args.accum_steps):
            x, y = data.next_batch(split='train')
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model_local(x, y)
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
            val_loss = evaluate(model_local, data, device, use_amp, max_batches=args.eval_batches)
            print(f"[val] step {step} loss {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.ckpt_path, model_local, optimizer, scaler, step, best_val)
                print(f"Saved best checkpoint (val {best_val:.6f}) to {args.ckpt_path}")
            if val_loss < args.target_loss:
                print(f"Target val loss {args.target_loss} reached at step {step} (val {val_loss:.6f}).")
                break

        if step % args.save_interval == 0:
            save_checkpoint(f"{args.ckpt_path}.step{step}", model_local, optimizer, scaler, step, best_val)

    pbar.close()
    # final save
    save_checkpoint(args.ckpt_path, model_local, optimizer, scaler, step, best_val)
    print(f"Training finished. Final step {step}, best val {best_val}")


def make_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--val_interval', type=int, default=500)
    p.add_argument('--save_interval', type=int, default=1000)
    p.add_argument('--eval_batches', type=int, default=32)
    p.add_argument('--target_loss', type=float, default=0.099999)
    p.add_argument('--ckpt_path', type=str, default='ckpt_best.pth')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--pretrained_model', type=str, default='gpt2')
    return p


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()
    # override defaults based on earlier values in file
    args.batch_size = args.batch_size or 4
    args.seq_len = args.seq_len or 32
    print(f"Training config: batch_size={args.batch_size}, seq_len={args.seq_len}, max_steps={args.max_steps}")
    train(args)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)