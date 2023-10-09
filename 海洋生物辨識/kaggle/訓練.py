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
'''
訓練資料的資料夾結構(各資料夾名稱可自訂):
data_folder
    Crabs
        *.jpg
    Dolphins
        *.jpg
    ...
'''
class CustomDataset(Dataset): # 繼承Dataset
    def __init__(self, data_dir, img_size, num_classes): # 定義物件時，會執行的初始化函式
        self.data_dir = data_dir # 設置訓練資料的位置
        self.img_size = img_size # 設置圖片大小
        self.num_classes = num_classes # 設置類的數量
        self.categories = ["Crabs", "Dolphin", "blue_ringed_octopus", "Sea Urchins", "Seahorse", "Turtle_Tortoise"]#, "Jelly Fish", "Lobster", "Nudibranchs", "Seal"]
        #self.categories = ["Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish", "Jelly Fish", "Lobster", "Nudibranchs", "Octopus"]#, "Otter", "Penguin", "Puffers", "Sea Rays", "Sea Urchins", "Seahorse", "Seal", "Sharks", "Shrimp", "Squid", "Starfish", "Turtle_Tortoise", "Whale"] # 訓練資料放置的資料夾名稱
        self.data, self.labels = self.load_data() # 把所有訓練圖片放入data,label放入labels

    def load_data(data_dir):
        images = []
        masks = []
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            for item in os.listdir(folder_path):
                if "mask" in item:
                    continue
                img_path = os.path.join(folder_path, item)
                mask_name = item.split('.')[0] + '_mask.png'
                mask_path = os.path.join(folder_path, mask_name)
                if not os.path.exists(mask_path):
                    print(f"Mask not found for image {img_path}")
                    continue
                
                # Try to load the image and mask
                try:
                    image = Image.open(img_path)
                    mask = Image.open(mask_path)
                except Exception as e:
                    print("Detailed error while loading:", str(e))
                    continue
                    
                images.append(img_path)
                masks.append(mask_path)
                
        return images, masks


    def __len__(self): # 獲取資料量
        return len(self.data)

    def __getitem__(self, index):
        """
        根據索引取得資料集中的一個樣本。

        :param index: 樣本的索引。
        :return: 返回樣本的圖片和標籤。
        """
        img = self.data[index]  # 取得圖片資料
        label = self.labels[index]  # 取得標籤資料
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV讀取的圖片是BGR順序，轉換為RGB順序
        img = transforms.ToTensor()(img)  # 將圖片轉換成PyTorch的Tensor格式
        label = torch.tensor(label, dtype=torch.long)  # 將標籤資料轉換為Long型態
        return img, label
 

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Batch Normalization after the first convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),  # Batch Normalization after the second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization after the third convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),  # Batch Normalization after the fifth convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 計算全連接層的輸入尺寸
        conv_output_size = input_shape[1] // 32

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size * conv_output_size * 1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

        # 初始化全連接層的權重
        nn.init.xavier_uniform_(self.fc_layers[1].weight)
        nn.init.xavier_uniform_(self.fc_layers[3].weight)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(num_epochs): # 跑num_epochs個epoch
        model.train() # 轉訓練模式
        # 準確度、損失初始化
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad() # 清除梯度
            outputs = model(inputs) # 把訓練圖片傳入模型，輸出輸出層值
            loss = criterion(outputs, labels) # 計算此次傳播的損失
            loss.backward() # 反向傳播
            optimizer.step() # 執行優化器，更新模型的參數
            
            train_loss += loss.item() # 更新訓練損失（Accumulate the training loss）
            _, predicted = torch.max(outputs.data, 1) # 計算預測值（Compute predictions）
            total_train += labels.size(0) # 累計訓練樣本數（Accumulate total training samples）
            correct_train += (predicted == labels).sum().item() # 累計正確預測數（Accumulate correct predictions）

        train_loss /= len(train_loader) # 將訓練的資料的loss做平均
        train_accuracy = correct_train / total_train # 將訓練的資料的accuracy做平均

        model.eval() # 轉評估模式
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad(): # 不進行反向傳播
            for inputs, labels in test_loader:
                outputs = model(inputs) # 把驗證圖片傳入模型，輸出輸出層值
                loss = criterion(outputs, labels) # 計算此次傳播的損失
                test_loss += loss.item() # 更新驗證損失
                _, predicted = torch.max(outputs.data, 1) # 計算預測值
                total_test += labels.size(0) # 累計驗證樣本數
                correct_test += (predicted == labels).sum().item() # 累計正確預測數

        test_loss /= len(test_loader) # 將驗證的資料的loss做平均
        test_accuracy = correct_test / total_test # 將驗證的資料的accuracy做平均
        
        scheduler.step(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 收集數據用於繪製曲線
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

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

# 設置數據目錄和影像大小
#data_dir = "dataset\\"  # 替換成包含"cat"和"dog"文件夾的數據集目錄
img_size = 224   # 替換成您想要的影像大小，例如128x128
num_classes = 6  # 海洋生物類別


# 載入數據集並預處理
# 創建 CustomDataset 對象
train_dataset = CustomDataset("train_processed\\", img_size, num_classes)
val_dataset = CustomDataset("test_processed\\", img_size, num_classes)
test_dataset = CustomDataset("val_processed\\", img_size, num_classes)
# 使用 DataLoader 封裝訓練集、驗證集和測試集
# DataLoader 用於將資料集封裝成一個可以迭代的物件，也就是可以用for來按照批次進行載入，進行訓練
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 以32張作批次處理，shuffle=True打亂資料
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# 建立模型
input_shape = (3, img_size, img_size)  # 假設圖像大小為img_size x img_size，且為RGB影像（3通道）
model = CNNModel(input_shape, num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 訓練模型並繪製曲線
num_epochs = 100
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 保存訓練好的模型
torch.save(model.state_dict(), "model_kaggle_6class_processed_epoch100_batchnorm.pth")

# 繪製混淆矩陣

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs) # 把測試圖片傳入模型，輸出輸出層值
        _, predicted = torch.max(outputs.data, 1) # 使用 torch.max() 函數計算出每個樣本的預測值，並將其儲存為 predicted
        # 將真實值和預測值附加到對應的列表中
        y_true.extend(labels.cpu().numpy()) 
        y_pred.extend(predicted.cpu().numpy())
        '''
        labels: 這是一個包含了一個批次中所有樣本的真實標籤的 PyTorch 張量。

        labels.cpu(): 將這個張量轉移到 CPU 上進行後續的處理。

        .numpy(): 將這個 PyTorch 張量轉換成 NumPy 陣列。

        y_true.extend(...): 將這個 NumPy 陣列中的元素添加到 y_true 這個列表的末尾。
        '''

# 畫confusion_matrix
cm = confusion_matrix(y_true, y_pred, normalize='true') # normalize='true':將confusion_matrix內的值歸一化
plt.figure(figsize=(num_classes,num_classes ))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')  # fmt='.2f' 顯示為浮點數，保留兩位小數
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show() # 顯示
