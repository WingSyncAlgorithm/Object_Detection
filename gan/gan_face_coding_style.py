import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
import torchvision.utils as vutils


class Config:
    # 將所有配置放在一個類別中，方便管理
    dataroot = "dataset\\data"
    batch_size = 100
    image_size = 128
    nz = 1000
    num_epochs = 100
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    num_classes = 1
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")

# 這裡保留您原本的 CustomDataset 和 CustomDataset2 類別


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class CustomDataset2(Dataset):
    def __init__(self, data_dir, img_size, num_classes):
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.categories = ["face"]
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            # 找出category在self.categories中的索引，ex:貓是0,狗是1,......
            label = self.categories.index(category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                if img is None:  # 檢查是否為非圖片文件，是則跳過
                    print("Error loading:", img_path)
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                data.append(img)
                labels.append(label)

        data = np.array(data)
        labels = np.array(labels)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): 樣本的索引

        Returns: 返回樣本的圖片和標籤
        """
        img = self.data[index]
        label = self.labels[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        label = torch.tensor(label, dtype=torch.long)  # 將標籤資料轉換為Long型態
        return img, label


def create_dataset(dataset_type, dataroot, image_size, num_classes):
    # 通過工廠方法創建數據集
    if dataset_type == 'custom':
        return CustomDataset2(dataroot, image_size, num_classes)
    # 可以在此添加其他數據集類型的創建邏輯
    else:
        raise ValueError("Unknown dataset type")


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        config = Config()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 4, 0, bias=False),
            nn.Tanh()
            # state size. 3 x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu,
                 input_shape=(3, 128, 128), num_classes=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        # print(x.size())
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(images, epoch):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    print(sqrtn)
    plt.figure(figsize=(10, 10))  # 可以根据需要调整大小
    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index + 1)
        # 转换图像格式从[channels, height, width]到[height, width, channels]
        image = np.transpose(image, (1, 2, 0))
        # 由于图像数据可能被归一化，需要调整到[0, 1]范围以正确显示
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴
    plt.savefig("Generator_epoch_{}.png".format(epoch))
    plt.show()


def update_network(model, optimizer, criterion, inputs, labels):
    """
    通用網絡更新器。
    :param model: 要更新的神經網絡模型。
    :param optimizer: 使用的優化器。
    :param criterion: 損失函數。
    :param inputs: 輸入數據。
    :param labels: 目標標籤。
    :param backward: 是否執行反向傳播。
    :return: 計算的損失和模型輸出。
    """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    return loss, outputs


def train(config, dataloader, generator, discriminator):
    device = config.device
    fixed_noise = torch.randn(16, config.nz, 1, 1, device=device)
    # 初始化 BCELoss 函數
    criterion = nn.BCELoss()

    # 建立真實和假的標籤變量
    real_label = 1.
    fake_label = 0.

    # 設定 Adam 優化器
    optimizerD = optim.Adam(discriminator.parameters(),
                            lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(),
                            lr=config.lr, betas=(config.beta1, 0.999))

    # 訓練過程中的追踪指標
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0

    print("開始訓練...")
    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # (1) 更新判別器網絡: maximize log(D(x)) + log(1 - D(G(z)))
            optimizerD.zero_grad()
            # 訓練全部真實的批次資料
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            D_x = output.mean().item()

            # 訓練全部假的批次資料
            noise = torch.randn(batch_size, config.nz, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1)
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            d_g_z1 = output.mean().item()
            loss_d_total = loss_d_real + loss_d_fake
            optimizerD.step()

            # (2) 更新生成器網絡: maximize log(D(G(z)))
            optimizerG.zero_grad()
            label.fill_(real_label)  # 生成器的假標籤是真的
            output = discriminator(fake_images).view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            d_g_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config.num_epochs, i, len(dataloader), loss_d_total.item(), loss_g.item(), D_x, d_g_z1, d_g_z2))

            g_losses.append(loss_g.item())
            d_losses.append(loss_d_total.item())

            # 檢查生成器的進度
            if (iters % 500 == 0) or ((epoch == config.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    show_images(fake[:16], epoch)
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

            iters += 1

    return generator, discriminator, g_losses, d_losses, img_list


def main():
    config = Config()
    dataset = create_dataset('custom', config.dataroot,
                             config.image_size, config.num_classes)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True)

    generator = Generator(config.ngpu).to(config.device)
    discriminator = Discriminator(config.ngpu).to(config.device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    train(config, dataloader, generator, discriminator)


if __name__ == "__main__":
    main()
