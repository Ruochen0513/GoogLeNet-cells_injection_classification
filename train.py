import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model2 import GoogLeNet

class DiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = ['injFail', 'injSuccess']
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for folder_name in os.listdir(class_dir):
                folder_path = os.path.join(class_dir, folder_name)
                if os.path.isdir(folder_path):
                    img1_path = os.path.join(folder_path, '1.jpg')
                    img8_path = os.path.join(folder_path, '8.jpg')
                    self.samples.append((img1_path, img8_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img8_path, label = self.samples[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img8 = Image.open(img8_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img8 = self.transform(img8)

        diff = img8 - img1

        return diff, label


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(data.to(device))
        loss = loss_function(logits, target.to(device))
        loss.backward()
        optimizer.step()
        print("  ",f'Train Epoch: {epoch} Loss: {loss.item():.6f}')


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data.to(device))
            val_loss += criterion(output, target.to(device)).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)

    return val_loss, accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = DiffDataset(root_dir='injection-dataset_student/train', transform=data_transform)
    val_dataset = DiffDataset(root_dir='injection-dataset_student/val', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = GoogLeNet(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 25
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print("Epoch", epoch,":training...")
        train(model, train_loader, optimizer, epoch, device)
        val_loss, accuracy = validate(model, val_loader, criterion, device)
        print("  ",f'Validation set: Average loss: {val_loss:.7f}, Accuracy: {accuracy:.0f}%')
        print("Epoch",epoch,":training finished.\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_googlenet_model.pth')


if __name__ == '__main__':
    main()
