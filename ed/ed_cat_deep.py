import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import cv2
import numpy as np
from model import SegNet3, SegNet, SegNet2, DeepSegNet, EncoderDecoderModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

class UAVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.colors = [
            (0, 0, 0),
            (0, 128, 0),
            (64, 0, 128),
            (64, 64, 0),
            (128, 0, 0),
            (128, 64, 128),
            (128, 128, 0),
            (192, 0, 192)
        ]
        self.image_paths = []
        self.label_paths = []
        self.size = 256

        seq_list = [dir_name for dir_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir_name))]

        for seq in seq_list:
            seq_image_dir = os.path.join(root_dir, seq, 'Images')
            seq_label_dir = os.path.join(root_dir, seq, 'Labels')
            self.image_paths.extend([os.path.join(seq_image_dir, f) for f in os.listdir(seq_image_dir) if f.endswith('.png')])
            self.label_paths.extend([os.path.join(seq_label_dir, f) for f in os.listdir(seq_label_dir) if f.endswith('.png')])

    def image_to_channels(self, img):
        data = np.array(img)
        channels = np.zeros(data.shape[:2] + (len(self.colors),), dtype=np.uint8)

        for i, color in enumerate(self.colors):
            mask = (data == color).all(axis=2)
            channels[mask, i] = 1

        # Return channels directly without converting to PIL Image
        return channels

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label_name = self.label_paths[idx]

        image = cv2.imread(img_name)
        label = cv2.imread(label_name)

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Resize both image and label to (256, 256)
        image = cv2.resize(image, (self.size, self.size))
        label = cv2.resize(label, (self.size, self.size))

        # Apply image_to_channels to the label
        label = self.image_to_channels(label)

        if self.transform:
            image = self.transform(image)
            # Convert label to tensor directly without using the provided transform
            label = torch.from_numpy(label).float()
            label = label.permute(2,0,1)
            #print(label.size())

        return image, label
    
    def __len__(self):
        return len(self.image_paths)




def accuracy(output, target):
    pred = (output > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / output.numel()

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=200):
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    model = model.to(device)  # Move the model to the device
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(inputs.size(),outputs.size())
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
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_accuracies.append(accuracy(outputs, labels))

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_accuracy = (sum(train_accuracies) / len(train_accuracies)).item()
        avg_val_accuracy = (sum(val_accuracies) / len(val_accuracies)).item()


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

        #torch.save(model.state_dict(), 'encoder_decoder_model_gpu.pth')
        model_path = os.path.join(save_dir, f'encoder_decoder_model_cat_size256_SegNet_batch1_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)


def main():
    size = 256
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Converting numpy array to PIL Image to apply transformations
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    train_dataset = UAVDataset(root_dir='uavid_v1.5_official_release_image/uavid_train', transform=transform)
    val_dataset = UAVDataset(root_dir='uavid_v1.5_official_release_image/uavid_val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SegNet(3,8)
    #model = DeepSegNet(3,8)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, val_loader, model, criterion, optimizer)

if __name__ == '__main__':
    main()
