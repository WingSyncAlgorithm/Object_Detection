import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
'''
訓練資料的資料夾結構(各資料夾名稱可自訂):
data_folder
    cat
        *.jpg
    dog
        *.jpg
    ...
'''
class CustomDataset(Dataset): # 繼承Dataset
    def __init__(self, data_dir, img_size, num_classes): # 定義物件時，會執行的初始化函式
        self.data_dir = data_dir # 設置訓練資料的位置
        self.img_size = img_size # 設置圖片大小
        self.num_classes = num_classes # 設置類的數量
        self.categories = ["cat", "dog"] # 訓練資料放置的資料夾名稱
        self.data, self.labels = self.load_data() # 把所有訓練圖片放入data,label放入labels

    def load_data(self):
        data = []
        labels = []
        for category in self.categories: # 跑遍所有類
            path = os.path.join(self.data_dir, category) # 把路徑加起來，得到各類資料夾的路徑
            label = self.categories.index(category) # 找出category在self.categories中的索引，ex:貓是0,狗是1
            for img_name in os.listdir(path): # 跑遍所有圖片
                img_path = os.path.join(path, img_name) # 獲取圖片的路徑
                print("Loading:", img_path)  # 為了除錯而印出圖像路徑
                img = cv2.imread(img_path) # 讀取圖片
                if img is None:
                    print("Error loading:", img_path)
                    continue  # 如果讀取失敗，則跳過該圖像
                img = cv2.resize(img, (self.img_size, self.img_size)) #將所有圖片都縮放成相同大小
                data.append(img) # 把圖片放進去
                labels.append(label) # 把label放進去

        data = np.array(data)
        labels = np.array(labels)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV讀取的圖片是BGR順序，轉換為RGB順序
        img = transforms.ToTensor()(img)  # 將圖片轉換成PyTorch的Tensor格式
        label = torch.tensor(label, dtype=torch.long)  # 將標籤資料轉換為Long型態
        return img, label


class CNNModel(nn.Module): # 定義class CNNModel，並繼承torch.nn.Module
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3通道(rgb)輸入,輸出32個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.ReLU(), # 負值變0
            nn.MaxPool2d(2), # 2*2方格取最大值

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32通道輸入上一層輸出的32張圖,輸出64個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.ReLU(), # 負值變0
            nn.MaxPool2d(2), # 2*2方格取最大值

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64通道輸入上一層輸出的64張圖,輸出128個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.ReLU(), # 負值變0
            nn.MaxPool2d(2) # 2*2方格取最大值
        )

        # 計算全連接層的輸入尺寸
        conv_output_size = input_shape[1] // 2 // 2 // 2  # 在三次池化後的影像尺寸，因為做了三次2*2的MaxPooling
        self.fc_layers = nn.Sequential( # 定義全連接層
            nn.Flatten(), # 把所有特徵圖，降成一維張量
            nn.Linear(conv_output_size * conv_output_size * 128, 64), # 接一層神經元，輸入通道數=128張特徵圖的總像素量，輸出64個通道
            nn.ReLU(), # 接讓負值變0的神經元，輸出與輸入輸量相同
            nn.Linear(64, num_classes), # 這是輸出層，將上一層輸出的64個輸出值，各類別可能性大小
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x) # 正向傳播經過卷積層
        x = self.fc_layers(x) # 正向傳播經過全連接層
        return x # 輸出輸出層的值


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct_test / total_test

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 收集數據用於繪製曲線
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # 繪製學習曲線和損失曲線
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

# 設置數據目錄和影像大小
data_dir = "..\\dataset\\"  # 替換成包含"cat"和"dog"文件夾的數據集目錄
img_size = 128   # 替換成您想要的影像大小，例如128x128
num_classes = 2  # 貓和狗兩個類別

# 載入數據集並預處理
dataset = CustomDataset(data_dir, img_size, num_classes)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 建立模型
input_shape = (3, img_size, img_size)  # 假設圖像大小為img_size x img_size，且為RGB影像（3通道）
model = CNNModel(input_shape, num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 訓練模型並繪製曲線
num_epochs = 10
model = train_model(model, data_loader, data_loader, criterion, optimizer, num_epochs)

# 保存訓練好的模型
torch.save(model.state_dict(), "model.pth")

# 載入測試集並繪製混淆矩陣
test_dataset = CustomDataset(data_dir, img_size, num_classes)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
