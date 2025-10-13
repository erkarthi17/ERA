"""
Quick test training script for EC2.
Tests the training pipeline with minimal epochs and reduced batch size.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main
import argparse


def test_training_quick():
    """
    Run a quick test training with minimal settings.
    """
    # Override sys.argv for testing
    original_argv = sys.argv.copy()
    
    # Set test arguments
    test_args = [
        'test_training.py',
        '--epochs', '2',              # Only 2 epochs
        '--batch-size', '64',         # Smaller batch size
        '--lr', '0.01',               # Lower learning rate
        '--data-root', '/data/imagenet'  # Update this path
    ]
    
    sys.argv = test_args
    
    print("="*80)
    print("QUICK TRAINING TEST")
    print("="*80)
    print("Settings:")
    print(f"  Epochs: 2")
    print(f"  Batch size: 64")
    print(f"  Learning rate: 0.01")
    print("="*80)
    print()
    
    try:
        # Run training
        main()
        print("\n" + "="*80)
        print("✅ Quick test completed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == '__main__':
    # You can also pass arguments directly
    parser = argparse.ArgumentParser(description='Quick test training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--data-root', type=str, required=True, help='Path to ImageNet data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUICK TRAINING TEST")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Data path: {args.data_root}")
    print("="*80)
    print()
    
    # Import and run main with test arguments
    import sys
    original_argv = sys.argv.copy()
    
    # Build command line arguments
    sys.argv = [
        'test_training.py',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--data-root', args.data_root
    ]
    
    try:
        from train import main
        main()
    finally:
        sys.argv = original_argv

