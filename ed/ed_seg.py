import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定義自定义數據集
class UAVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        seq_list = [dir_name for dir_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir_name))]

        for seq in seq_list:
            seq_image_dir = os.path.join(root_dir, seq, 'Images')
            seq_label_dir = os.path.join(root_dir, seq, 'Labels')
            self.image_paths.extend([os.path.join(seq_image_dir, f) for f in os.listdir(seq_image_dir) if f.endswith('.png')])
            self.label_paths.extend([os.path.join(seq_label_dir, f) for f in os.listdir(seq_label_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label_name = self.label_paths[idx]
        image = Image.open(img_name)
        label = Image.open(label_name)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def accuracy(output, target):
    pred = (output > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / output.numel()

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append(accuracy(outputs, labels))

        model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_accuracies.append(accuracy(outputs, labels))

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

        all_train_losses.append(avg_train_loss)
        all_val_losses.append(avg_val_loss)
        all_train_accuracies.append(avg_train_accuracy)
        all_val_accuracies.append(avg_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Train Accuracy: {avg_train_accuracy:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}')

        # 繪製損失曲線
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(all_train_losses, '-x', label='Train')
        plt.plot(all_val_losses, '-o', label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss vs. No. of epochs')

        # 繪製準確度曲線
        plt.subplot(1, 2, 2)
        plt.plot(all_train_accuracies, '-x', label='Train')
        plt.plot(all_val_accuracies, '-o', label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.title('Accuracy vs. No. of epochs')
        plt.show()

    torch.save(model.state_dict(), 'encoder_decoder_model.pth')

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = UAVDataset(root_dir='uavid_v1.5_official_release_image/uavid_train', transform=transform)
    val_dataset = UAVDataset(root_dir='uavid_v1.5_official_release_image/uavid_val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = EncoderDecoderModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, val_loader, model, criterion, optimizer)

if __name__ == '__main__':
    main()
