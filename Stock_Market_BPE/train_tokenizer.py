import sys
import io
from tokenizer import StockBPE
from pathlib import Path
import time

# Fix console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def train_and_verify():
    data_path = Path("stock_corpus.txt")
    
    if not data_path.exists():
        print("❌ Error: stock_corpus.txt not found!")
        print("Please run 'python download_stock_data.py' first")
        return
    
    # Check file size
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Read data
    print(f"Reading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Data size: {len(text):,} characters")
    print(f"Sample data:\n{text[:200]}...\n")
    
    # Initialize tokenizer
    tokenizer = StockBPE()
    vocab_size = 5500  # Target > 5000
    
    print(f"Training tokenizer with vocab size {vocab_size}...")
    print("This should take 2-5 minutes...\n")
    
    start_time = time.time()
    tokenizer.train(text, vocab_size)
    elapsed = time.time() - start_time
    
    print(f"\nTraining took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save("stock_bpe")
    print("✓ Saved to: stock_bpe.merges and stock_bpe.vocab")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    # Calculate compression ratio
    ratio = tokenizer.calculate_compression_ratio(text)
    print(f"Compression Ratio: {ratio:.2f}")
    
    vocab_len = len(tokenizer.vocab)
    print(f"Vocabulary Size: {vocab_len}")
    
    # Check requirements
    print("\n" + "="*70)
    if vocab_len > 5000 and ratio >= 3:
        print("✅ SUCCESS: Requirements met!")
        print(f"  ✓ Vocabulary size: {vocab_len} (required: > 5000)")
        print(f"  ✓ Compression ratio: {ratio:.2f} (required: >= 3.0)")
    else:
        print("⚠️  WARNING: Requirements NOT fully met.")
        if vocab_len <= 5000:
            print(f"  ✗ Vocabulary size: {vocab_len} (required: > 5000)")
        else:
            print(f"  ✓ Vocabulary size: {vocab_len} (required: > 5000)")
        
        if ratio < 3:
            print(f"  ✗ Compression ratio: {ratio:.2f} (required: >= 3.0)")
        else:
            print(f"  ✓ Compression ratio: {ratio:.2f} (required: >= 3.0)")
    print("="*70)
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    
    # Test with a sample stock data line
    sample_lines = text.split('\n')[:3]
    for sample_text in sample_lines:
        if sample_text.strip():
            encoded = tokenizer.encode(sample_text)
            decoded = tokenizer.decode(encoded)
            
            print(f"\nOriginal: {sample_text}")
            print(f"Encoded:  {encoded[:20]}... ({len(encoded)} tokens)")
            print(f"Decoded:  {decoded}")
            print(f"Match: {'✓' if sample_text == decoded else '✗'}")
            
            assert sample_text == decoded, "Encoding/decoding mismatch!"
    
    print("\n✅ All encoding/decoding tests passed!")
    
    # Show some interesting statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total characters: {len(text):,}")
    print(f"Total lines: {len(text.split(chr(10))):,}")
    print(f"Vocabulary size: {vocab_len:,}")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Original size: {len(text.encode('utf-8')):,} bytes")
    print(f"Compressed size: {len(tokenizer.encode(text)):,} tokens")
    print("="*70)

if __name__ == "__main__":
    train_and_verify()
