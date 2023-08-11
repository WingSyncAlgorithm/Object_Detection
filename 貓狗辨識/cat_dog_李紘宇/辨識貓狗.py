import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 計算全連接層的輸入尺寸
        conv_output_size = input_shape[1] // 2 // 2 // 2  # 在三次池化後的影像尺寸
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size * conv_output_size * 128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 設置影像大小和類別數量
img_size = 128  # 與訓練時相同的圖片尺寸
num_classes = 2  # 貓和狗兩個類別

# 建立模型並載入訓練好的權重
model = CNNModel(input_shape=(3, img_size, img_size), num_classes=num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # 設定為評估模式，不使用dropout等

# 讀取待預測的圖片
image_path = "test.jpg"

# 讀取並預處理圖片
img = cv2.imread(image_path)
img = cv2.resize(img, (img_size, img_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV讀取的圖片是BGR順序，轉換為RGB順序
img = transforms.ToTensor()(img)  # 將圖片轉換成PyTorch的Tensor格式
img = img.unsqueeze(0)  # 增加一個batch的維度

# 使用模型進行預測
with torch.no_grad():
    outputs = model(img)
    _, predicted_class = torch.max(outputs, 1)

# 獲得預測結果
if predicted_class.item() == 0:
    print("這是一隻貓")
else:
    print("這是一隻狗")
