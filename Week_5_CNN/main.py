from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary  # pip install torchsummary


# Define the CNN architecture
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         # Input Block - 1x28x28 (RF: 1)
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=1, bias=False),  # 8x28x28 | RF: 3
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )

#         # Block 2
#         self.convblock2 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, padding=1, bias=False),  # 16x28x28 | RF: 5
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         self.pool1 = nn.MaxPool2d(2, 2)  # 16x14x14 | RF: 6

#         # Block 3
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(16, 16, 3, padding=1, bias=False),  # 16x14x14 | RF: 10
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )

#         # Block 4
#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(16, 16, 3, padding=1, bias=False),  # 16x14x14 | RF: 14
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         self.pool2 = nn.MaxPool2d(2, 2)  # 16x7x7 | RF: 16

#         # Block 5 + Dropout
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(16, 16, 3, padding=1, bias=False),  # 16x7x7 | RF: 24
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Dropout(0.1)  # Regularization
#         )

#         # Output Block
#         self.convblock6 = nn.Conv2d(16, 10, 1, bias=False)  # 10x7x7 | RF: 24
#         self.gap = nn.AdaptiveAvgPool2d(1)  # 10x1x1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# CUDA setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Model summary
model = Net().to(device)
print("Model Summary:")
summary(model, input_size=(1, 28, 28))
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Data setup
torch.manual_seed(1)
batch_size = 128

train_transforms = transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0), fill=(0,)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
train_set, val_set = torch.utils.data.random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {} # Increased num_workers
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)


# Training
def train(model, device, loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(loader)
    correct = 0 # Initialize correct predictions for training
    processed = 0 # Initialize total processed samples for training
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update progress bar description with loss and training accuracy
        pbar.set_description(f'Epoch {epoch} Loss={loss.item():.4f} Accuracy={100.*correct/processed:.2f}%')


# Validation
def validate(model, device, loader):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return val_loss, accuracy


# Main
if __name__ == '__main__':
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    epochs = 19
    scheduler = OneCycleLR(optimizer, max_lr=0.005,
                           steps_per_epoch=len(train_loader),
                           epochs=epochs, pct_start=0.2)

    best_acc = 0
    print(f"Training for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        _, acc = validate(model, device, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_mnist_cnn.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    print(f"\nTraining finished. Best Validation Accuracy: {best_acc:.2f}%")
    print("--- Detailed Parameter Count for README ---")
    summary(model, input_size=(1, 28, 28))
    print("-----------------------------------------")