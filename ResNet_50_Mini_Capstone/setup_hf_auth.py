"""
Hugging Face Authentication Setup Script
This script helps you set up authentication for Hugging Face datasets.
"""

import os
import sys
from huggingface_hub import login, whoami, HfApi

def setup_authentication():
    """Setup Hugging Face authentication."""
    print("="*60)
    print("Hugging Face Authentication Setup")
    print("="*60)
    
    # Check if already logged in
    try:
        user_info = whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not currently logged in to Hugging Face")
    
    print("\n🔑 Authentication Options:")
    print("1. Enter token directly")
    print("2. Use environment variable")
    print("3. Skip authentication (for public datasets only)")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        return setup_with_token()
    elif choice == "2":
        return setup_with_env_var()
    elif choice == "3":
        print("⚠️  Skipping authentication. You can only access public datasets.")
        return True
    else:
        print("❌ Invalid choice")
        return False

def setup_with_token():
    """Setup authentication with direct token input."""
    print("\n📝 Getting your Hugging Face token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Copy the token and paste it below")
    
    token = input("\nEnter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    try:
        # Login with token
        login(token=token)
        
        # Verify login
        user_info = whoami()
        print(f"✅ Successfully logged in as: {user_info['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def setup_with_env_var():
    """Setup authentication with environment variable."""
    print("\n🔧 Environment Variable Setup:")
    print("You can set the token as an environment variable:")
    print("\nWindows PowerShell:")
    print('$env:HUGGINGFACE_HUB_TOKEN = "your_token_here"')
    print("\nWindows CMD:")
    print('set HUGGINGFACE_HUB_TOKEN=your_token_here')
    print("\nOr add it to your system environment variables permanently.")
    
    # Check if already set
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        print(f"\n✅ Found token in environment variable")
        try:
            # Test the token
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"✅ Token is valid. Logged in as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Token is invalid: {e}")
            return False
    else:
        print("\n❌ No token found in environment variables")
        print("Please set HUGGINGFACE_HUB_TOKEN environment variable first")
        return False

def test_imagenet_access():
    """Test access to ImageNet dataset."""
    print("\n" + "="*60)
    print("Testing ImageNet Access")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        print("🔄 Attempting to load ImageNet dataset...")
        # Try to load a tiny subset first
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train[:10]")
        print("✅ Successfully accessed ImageNet dataset!")
        print(f"📊 Loaded {len(dataset)} samples for testing")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "gated dataset" in error_msg.lower():
            print("❌ ImageNet is a gated dataset. You need:")
            print("   1. A Hugging Face account")
            print("   2. Request access to ImageNet at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print("   3. Wait for approval (usually instant for most users)")
        elif "authentication" in error_msg.lower():
            print("❌ Authentication issue. Please check your token.")
        else:
            print(f"❌ Error accessing ImageNet: {e}")
        return False

def show_alternatives():
    """Show alternative approaches if ImageNet access fails."""
    print("\n" + "="*60)
    print("Alternative Approaches")
    print("="*60)
    
    print("\n🔄 If you can't access ImageNet, try these alternatives:")
    
    print("\n1. 📦 Use CIFAR-10 for testing (public dataset):")
    print("   python demo_hf_concept.py")
    
    print("\n2. 🎯 Use a different ImageNet-like dataset:")
    print("   - imagenet-1k-tiny (smaller version)")
    print("   - food101 (food classification)")
    print("   - oxford_flowers102 (flower classification)")
    
    print("\n3. 🛠️  Modify the code to use a different dataset:")
    print("   Edit data_hf.py and change 'imagenet-1k' to another dataset")
    
    print("\n4. 📁 Use local ImageNet files (original approach):")
    print("   Use the original data.py and train.py files")

def main():
    """Main setup function."""
    print("🚀 Hugging Face Setup Assistant")
    
    # Step 1: Authentication
    auth_success = setup_authentication()
    
    if not auth_success:
        print("\n❌ Authentication setup failed")
        show_alternatives()
        return
    
    # Step 2: Test ImageNet access
    imagenet_success = test_imagenet_access()
    
    if imagenet_success:
        print("\n🎉 Setup Complete!")
        print("\nYou can now run:")
        print("   python test_hf_setup.py")
        print("   python train_hf.py --train-subset 1000 --epochs 1")
    else:
        print("\n⚠️  ImageNet access failed, but authentication is working")
        show_alternatives()

if __name__ == "__main__":
    main()
