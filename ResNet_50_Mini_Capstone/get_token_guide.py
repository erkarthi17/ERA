"""
Simple guide to get your Hugging Face token.
"""

def main():
    print("="*60)
    print("How to Get Your Hugging Face Token")
    print("="*60)
    
    print("\nStep-by-Step Instructions:")
    print("\n1. Go to: https://huggingface.co/settings/tokens")
    print("   (You need to be logged into your Hugging Face account)")
    
    print("\n2. Click 'New token' button")
    print("   - Name: 'ImageNet Access' (or any name you prefer)")
    print("   - Type: 'Read' (this is sufficient for downloading datasets)")
    
    print("\n3. Copy the generated token")
    print("   - It will look like: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("   - Keep this token secure!")
    
    print("\n4. Set up authentication:")
    print("   Option A - Command line:")
    print("   > huggingface-cli login")
    print("   (Then paste your token when prompted)")
    
    print("\n   Option B - Environment variable (Windows PowerShell):")
    print('   > $env:HUGGINGFACE_HUB_TOKEN = "your_token_here"')
    
    print("\n   Option C - Environment variable (Windows CMD):")
    print('   > set HUGGINGFACE_HUB_TOKEN=your_token_here')
    
    print("\n5. Test your setup:")
    print("   > python setup_hf_auth.py")
    
    print("\n" + "="*60)
    print("ImageNet Access Requirements")
    print("="*60)
    
    print("\n🎯 To access ImageNet-1k dataset:")
    print("1. You need a Hugging Face account (free)")
    print("2. Request access at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    print("3. Usually approved instantly for most users")
    print("4. Then use the token above to authenticate")
    
    print("\n🔄 Alternative if ImageNet access is denied:")
    print("- Use CIFAR-10 for testing (public dataset)")
    print("- Try other datasets like Food-101 or Oxford Flowers")
    print("- Use the original local ImageNet files")
    
    print("\n💡 Quick Test (no authentication needed):")
    print("> python demo_hf_concept.py")
    print("(This uses CIFAR-10 which is public)")

if __name__ == "__main__":
    main()
