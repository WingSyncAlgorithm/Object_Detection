import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.models as models
from torch.optim import Adam

# Dataset
class UAVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dirs = [os.path.join(root_dir, seq, "Images") for seq in os.listdir(root_dir)]
        self.label_dirs = [os.path.join(root_dir, seq, "Labels") for seq in os.listdir(root_dir) if os.path.exists(os.path.join(root_dir, seq, "Labels"))]
        self.image_files = []
        self.label_files = []
        
        for img_dir, lbl_dir in zip(self.image_dirs, self.label_dirs):
            print(root_dir)
            img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
            lbl_files = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)])
            
            # Only keep images that have a corresponding label and vice versa
            common_files = set([os.path.basename(f) for f in img_files]) & set([os.path.basename(f) for f in lbl_files])
            self.image_files.extend([f for f in img_files if os.path.basename(f) in common_files])
            self.label_files.extend([f for f in lbl_files if os.path.basename(f) in common_files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        label = Image.open(self.label_files[idx])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# Model Architecture

class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()
        self.encoder = models.vgg16_bn(pretrained=True).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = nn.Sigmoid()(self.deconv3(x))
        return x

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training Function (similar to the one you provided)
def compute_accuracy(pred, target):
    """
    Compute pixel-wise accuracy for segmentation.
    """
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / (pred.numel())

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for inputs, labels in train_loader:
            print(inputs.size())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += compute_accuracy(outputs, inputs)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
                test_accuracy += compute_accuracy(outputs, inputs)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
        

        # 绘制学习曲线和损失曲线图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 2), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 2), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epoch + 2), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model

# Assuming the rest of the code remains unchanged, 
# you would call train_model_with_accuracy instead of train_model to train your model and plot accuracy curves.


# Main
num_classes = 8

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_dataset = UAVDataset("uavid_v1.5_official_release_image\\uavid_train", transform=transform)
test_dataset = UAVDataset("uavid_v1.5_official_release_image\\uavid_val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#model = AutoEncoderModel()
model = SegNet(input_channels=3, output_channels=num_classes)  # Assume num_classes is the number of classes for segmentation

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = Adam(model.parameters(), lr=0.001)

model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)

torch.save(model.state_dict(), "autoencoder_model.pth")
