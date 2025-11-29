import regex as re
import json
from tqdm import tqdm

class StockBPE:
    """BPE Tokenizer optimized for stock market time-series data"""
    
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        # OPTIMIZATION: Treat the entire line as a single chunk to allow merging 
        # labels with delimiters (e.g., "OPEN" + ":" -> "OPEN:")
        self.pattern = re.compile(r'[^\n]+|\n')
    
    def get_stats(self, ids):
        """Count frequency of adjacent pairs"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """Merge all occurrences of a pair"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text, vocab_size, verbose=True):
        """Train BPE on stock market data"""
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Pre-tokenize using pattern
        text_chunks = re.findall(self.pattern, text)
        
        # Convert to UTF-8 bytes
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        # Training loop with progress bar
        for i in tqdm(range(num_merges), desc="Training Stock BPE", unit="merge"):
            stats = {}
            for chunk_ids in ids:
                chunk_stats = self.get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] = stats.get(pair, 0) + count
            
            if not stats:
                print(f"\nNo more pairs to merge. Stopping at {i} merges.")
                break
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Apply merge
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            self.merges[pair] = idx
        
        # Build vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        print(f"\nTraining complete. Final vocab size: {len(self.vocab)}")
    
    def encode(self, text):
        """Encode text to token IDs"""
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = self.get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = self.merge(chunk_ids, pair, idx)
            ids.extend(chunk_ids)
        return ids
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
    
    def save(self, prefix):
        """Save tokenizer to files"""
        # Save merges
        with open(f"{prefix}.merges", "w", encoding="utf-8") as f:
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")
        
        # Save vocab
        vocab_str = {idx: token.decode("utf-8", errors="replace") 
                     for idx, token in self.vocab.items()}
        with open(f"{prefix}.vocab", "w", encoding="utf-8") as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    
    def load(self, prefix):
        """Load tokenizer from files"""
        self.merges = {}
        with open(f"{prefix}.merges", "r", encoding="utf-8") as f:
            for line in f:
                p0, p1, idx = map(int, line.split())
                self.merges[(p0, p1)] = idx
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def calculate_compression_ratio(self, text):
        """Calculate compression ratio"""
        encoded = self.encode(text)
        original_bytes = len(text.encode("utf-8"))
        compressed_tokens = len(encoded)
        return original_bytes / compressed_tokens if compressed_tokens > 0 else 0
