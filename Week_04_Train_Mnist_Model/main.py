import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
import os
import argparse

# ----------------------------
# Argument Parser
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="MNIST Training Script")

    parser.add_argument("--batch-size", type=int, default=512, help="input batch size for training (default: 512)")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--step-size", type=int, default=15, help="StepLR step size (default: 15)")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma decay (default: 0.1)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")

    return parser.parse_args()

# ----------------------------
# Model Definition
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)  # corrected size
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1)  # auto flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x
    
# --------------------------------------------
# A smaller model with fewer parameters
# --------------------------------------------
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, padding=1)     # 200 params
        self.bn1 = nn.BatchNorm2d(20)                   # 40

        self.conv2 = nn.Conv2d(20, 28, 3, padding=1)    # 5,048 params
        self.bn2 = nn.BatchNorm2d(28)                   # 56

        self.conv3 = nn.Conv2d(28, 64, 3, padding=1)    # 16,192 params
        self.bn3 = nn.BatchNorm2d(64)                   # 128

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)                     # 650

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
# Total params: ~22,514
    
# ----------------------------
# Training & Testing
# ----------------------------
train_losses, test_losses, train_acc, test_acc = [], [], [], []

def GetCorrectPredCount(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader, leave=False)
    correct, processed, train_loss = 0, 0, 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, target)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(f"Train: Loss={loss.item():0.4f} Accuracy={100*correct/processed:0.2f}")

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(f"Test: Average loss={test_loss:.4f}, Accuracy={correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset):.2f}%)")

# ----------------------------
# Main
# ----------------------------
def main():
    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("CUDA Available?", use_cuda)

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model, optimizer, scheduler
    # model = Net().to(device)
    model = TinyNet().to(device)  # Use TinyNet instead of Net for fewer parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    # Save results
    os.makedirs("results", exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses); axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc); axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses); axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc); axs[1, 1].set_title("Test Accuracy")
    plt.savefig("results/training_curves.png")

    with open("results/model_summary.txt", "w") as f:
        from contextlib import redirect_stdout
        with redirect_stdout(f):
            summary(model, input_size=(1, 28, 28))

    print("✅ Training complete. Results saved in 'results/' folder.")

if __name__ == "__main__":
    main()