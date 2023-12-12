import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義生成器
class Generator(nn.Module):
    """
    生成器模型
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 定義判別器
class Discriminator(nn.Module):
    """
    判別器模型
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 定義超參數
batch_size = 64
learning_rate = 0.0002
epochs = 100

# 載入 MNIST 數據集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: cv2.GaussianBlur(x.numpy(), (5, 5), 0).reshape(28, 28, 1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 初始化生成器和判別器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(epochs):
    for i, (blurred_images, _) in enumerate(train_loader):
        # 將模糊圖片轉換為向量
        blurred_images = blurred_images.view(-1, 784).to(device)

        # 訓練生成器
        optimizer_G.zero_grad()
        # 直接將生成器的輸入設置為模糊的圖像
        fake_images = generator(blurred_images)
        outputs = discriminator(fake_images)

        # 計算損失並更新生成器
        loss_G = criterion(outputs, torch.ones_like(outputs))
        loss_G.backward()
        optimizer_G.step()

        # 打印訓練信息
        if i % 100 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [G loss: %.4f]' %
                  (epoch, epochs, i, len(train_loader), loss_G.item()))

# 選擇一個糊的圖像
blurred_image, _ = next(iter(train_loader))
blurred_image = blurred_image[0].numpy().reshape(28, 28)

# 顯示糊的圖像
plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')

# 使用生成器生成預測的圖像
generated_image = generator(blurred_images)
generated_image_np = generated_image.detach().cpu().numpy()

# 調整生成的圖像大小為28*28
generated_image_np = cv2.resize(generated_image_np, (28, 28), interpolation=cv2.INTER_LINEAR)

# 顯示預測的圖像
plt.subplot(1, 2, 2)
plt.imshow(generated_image_np.squeeze(), cmap='gray')  # 使用squeeze()去除多餘的維度
plt.title('Generated Image')

plt.show()
