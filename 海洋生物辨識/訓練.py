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
        self.categories = ["Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish", "Jelly Fish", "Lobster", "Nudibranchs", "Octopus", "Otter", "Penguin", "Puffers", "Sea Rays", "Sea Urchins", "Seahorse", "Seal", "Sharks", "Shrimp", "Squid", "Starfish", "Turtle_Tortoise", "Whale"] # 訓練資料放置的資料夾名稱
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
data_dir = "dataset\\"  # 替換成包含"cat"和"dog"文件夾的數據集目錄
img_size = 128   # 替換成您想要的影像大小，例如128x128
num_classes = 23  # 貓和狗兩個類別


# 載入數據集並預處理
# 創建 CustomDataset 對象
train_dataset = CustomDataset("train\\", img_size, num_classes)
val_dataset = CustomDataset("test\\", img_size, num_classes)
test_dataset = CustomDataset("val\\", img_size, num_classes)
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
optimizer = optim.Adam(model.parameters())

# 訓練模型並繪製曲線
num_epochs = 10
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 保存訓練好的模型
torch.save(model.state_dict(), "model.pth")

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
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')  # fmt='.2f' 顯示為浮點數，保留兩位小數
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show() # 顯示
