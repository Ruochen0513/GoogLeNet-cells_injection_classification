import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model2 import GoogLeNet

class PredictDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                img1_path = os.path.join(folder_path, '1.jpg')
                img8_path = os.path.join(folder_path, '8.jpg')
                self.samples.append((img1_path, img8_path, folder_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img8_path, folder_name = self.samples[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img8 = Image.open(img8_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img8 = self.transform(img8)

        diff = img8 - img1

        return diff, folder_name

def predict(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for data, folder_name in data_loader:
            data = data.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1, keepdim=True)
            results.append((folder_name[0], pred.item()))
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    predict_dataset = PredictDataset(root_dir='injection-dataset_student/test', transform=data_transform)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=4)


    model = GoogLeNet(num_classes=2).to(device)
    model.load_state_dict(torch.load('best_googlenet_model.pth'))


    results = predict(model, predict_loader, device)

    for folder_name, pred in results:
        print(f'Folder: {folder_name}, Prediction: {"Success or 1" if pred == 1 else "Fail or 0"}')

if __name__ == '__main__':
    main()
