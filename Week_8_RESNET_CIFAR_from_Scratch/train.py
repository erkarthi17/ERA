import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import time
from tqdm import tqdm

from model import ResNet44V2
from data.transforms import get_transforms
from utils import accuracy, save_checkpoint, log_epoch
from config import config

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    print(f"📊 Configuration: {config}")
    print("=" * 60)
    
    train_transform, test_transform = get_transforms()

    print("📥 Loading CIFAR-100 dataset...")
    train_set = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    val_set = CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    print(f"📈 Dataset loaded: {len(train_set)} training samples, {len(val_set)} validation samples")

    model = ResNet44V2(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc = 0.0
    total_start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [Train]", 
                         leave=False, ncols=100)
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            correct += acc1 * targets.size(0) / 100
            total += targets.size(0)
            
            # Update progress bar with current metrics
            current_acc = 100.0 * correct / total
            current_loss = total_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        train_acc = 100.0 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation phase with progress bar
        model.eval()
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config['epochs']} [Val]", 
                       leave=False, ncols=100)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                correct += acc1 * targets.size(0) / 100
                total += targets.size(0)
                
                # Update validation progress bar
                current_val_acc = 100.0 * correct / total
                val_pbar.set_postfix({'Acc': f'{current_val_acc:.2f}%'})

        val_acc = 100.0 * correct / total
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time
        remaining_time = (elapsed_time / epoch) * (config["epochs"] - epoch)

        log_epoch(epoch, train_acc, val_acc, train_loss, config["log_file"])

        # Enhanced status output
        status_icon = "🆕" if val_acc > best_acc else "📊"
        print(f"\n{status_icon} Epoch {epoch:3d}/{config['epochs']} | "
              f"Train: {train_acc:6.2f}% ({train_loss:.4f}) | "
              f"Val: {val_acc:6.2f}% | "
              f"Time: {epoch_time:5.1f}s | "
              f"ETA: {remaining_time/60:5.1f}m | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, config["checkpoint_path"])
            print(f"     ✅ New best model saved! Val Acc: {val_acc:.2f}%")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()