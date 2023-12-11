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
    """
    將所有配置放在一個類別中，方便管理
    """
    dataroot = "dataset\\data_Crabs"
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


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化CustomDataset類的實例
        Args:
            root_dir (str): 包含圖像文件的根目錄路徑
            transform (callable, optional): 用於進行圖像轉換的可調用對象。默認為None，表示不進行轉換
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        """
        返回數據集中的圖像數量
        Returns:
            int: 數據集中的圖像數量
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        根據提供的索引從數據集中獲取一個圖像
        Args:
            idx (int): 圖像的索引
        Returns:
            image: 該索引的圖像
        """
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class CustomDataset2(Dataset):
    def __init__(self, data_dir, img_size, num_classes):
        """
        初始化CustomDataset2類的實例。
        Args:
            data_dir (str): 包含圖像文件夾的根目錄路徑。
            img_size (int): 圖像在加載時應調整到的目標大小。
            num_classes (int): 數據集中不同類別的數量。
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.categories = ["Crabs"]  # 目前只有一個類別
        self.data, self.labels = self.load_data()  # 加載數據和對應的標籤

    def load_data(self):
        """
        從指定目錄加載圖像數據和標籤。
        Returns:
            tuple: 包含兩個numpy陣列，分別是加載的圖像數據和對應的標籤。
        """
        data = []
        labels = []
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            label = self.categories.index(category)  # 獲取類別對應的索引作為標籤
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print("Error loading:", img_path)
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                data.append(img)
                labels.append(label)
        data = np.array(data)
        labels = np.array(labels)
        return data, labels

    def __len__(self):
        """
        返回數據集中的樣本數量。
        Returns:
            int: 數據集中的樣本數量。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根據提供的索引從數據集中獲取一個樣本及其標籤。
        Args:
            index (int): 樣本的索引。
        Returns:
            tuple: 包含一個轉換後的圖像張量和對應的標籤。
        """
        img = self.data[index]
        label = self.labels[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


def create_dataset(dataset_type, dataroot, image_size, num_classes):
    """
    根據指定的數據集類型創建數據集。
    Args:
        dataset_type (str): 要創建的數據集的類型。
        dataroot (str): 數據集的根目錄路徑。
        image_size (int): 圖像大小。
        num_classes (int): 分類的數量。
    Returns:
        Dataset: 創建的數據集對象。
    Raises:
        ValueError: 如果提供的數據集類型不被識別。
    """
    if dataset_type == 'custom':
        return CustomDataset2(dataroot, image_size, num_classes)
    else:
        raise ValueError("Unknown dataset type")


class Generator(nn.Module):
    def __init__(self, ngpu):
        """
        初始化生成器模型。
        Args:
            ngpu (int): 使用的GPU數量。
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        config = Config()
        self.main = nn.Sequential(
            # 第一層反卷積，將輸入的噪聲向量轉換為特徵圖。
            nn.ConvTranspose2d(config.nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 後續層逐漸放大特徵圖並減少深度。
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 4, 0, bias=False),
            nn.Tanh()  # 最後一層使用Tanh函數，將輸出壓縮到[-1, 1]範圍內。
        )

    def forward(self, input):
        """
        通過模型前向傳播輸入數據。
        Args:
            input (Tensor): 輸入的噪聲向量。
        Returns:
            Tensor: 生成的圖像數據。
        """
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, input_shape=(3, 128, 128), num_classes=1):
        """
        初始化判别器模型。
        Args:
            ngpu (int): 使用的GPU数量。
            input_shape (tuple, optional): 输入图像的形状。默认为(3, 128, 128)。
            num_classes (int, optional): 输出的类别数，对于二分类问题通常为1。默认为1。
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # 卷积层序列，用于从输入图像中提取特征。
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

        # 计算全连接层的输入大小。
        conv_output_size = input_shape[1] // 2 // 2 // 2  # 经过三次下采样后的图像尺寸。

        # 全连接层序列，用于最终的分类判断。
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size * conv_output_size * 128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # 使用Sigmoid函数来输出二分类问题的概率。
            # nn.Softmax(dim=1)  # 如果是多分类问题，可以使用Softmax函数。
        )

    def forward(self, x):
        """
        通过模型进行前向传播。
        Args:
            x (Tensor): 输入的图像数据。
        Returns:
            Tensor: 判别器的输出结果。
        """
        x = self.conv_layers(x)  # 将输入数据传递通过卷积层。
        x = self.fc_layers(x)  # 经过全连接层得到最终结果。
        x = x.unsqueeze(-1).unsqueeze(-1)  # 调整输出维度。
        return x


def weights_init(m):
    """
    對模型中的權重進行初始化。
    Args:
        m (nn.Module): PyTorch網絡中的一個模塊。
    """
    classname = m.__class__.__name__  # 獲取模塊的類名

    # 如果是卷積層（'Conv'在類名中），初始化其權重為正態分佈。
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    # 如果是批量歸一化層（'BatchNorm'在類名中），初始化權重為正態分佈，偏置為0。
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_images(images, epoch):
    """
    展示一批圖像。
    Args:
        images (Tensor): 包含多個圖像的張量。
        epoch (int): 當前的訓練周期，用於標記保存的圖像文件。
    """
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))  # 計算每邊應該展示的圖像數量
    print(sqrtn)
    plt.figure(figsize=(10, 10))  # 根據需要調整圖像展示的大小

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index + 1)  # 為每個圖像創建一個子圖
        # 轉換圖像格式從[channels, height, width]到[height, width, channels]
        image = np.transpose(image, (1, 2, 0))
        # 由於圖像數據可能被歸一化，需要調整到[0, 1]範圍以正確顯示
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)
        plt.axis('off')  # 關閉座標軸

    plt.savefig("Generator_epoch_{}.png".format(epoch))  # 保存圖像
    plt.show()  # 顯示圖像


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
    """
    訓練生成對抗網絡。
    Args:
        config (Config): 包含訓練配置的對象。
        dataloader (DataLoader): 數據加載器，用於提供訓練數據。
        generator (nn.Module): 生成器網絡。
        discriminator (nn.Module): 判別器網絡。
    Returns:
        tuple: 包含訓練後的生成器和判別器，以及生成器和判別器的損失列表。
    """
    device = config.device
    fixed_noise = torch.randn(16, config.nz, 1, 1, device=device)
    criterion = nn.BCELoss()  # 二元交叉熵損失

    real_label = 1.
    fake_label = 0.

    # 設置兩個網絡的優化器
    optimizerD = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    img_list, g_losses, d_losses = [], [], []
    iters = 0

    print("開始訓練...")
    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # 更新判別器網絡
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()

            noise = torch.randn(batch_size, config.nz, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1)
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            loss_d_total = loss_d_real + loss_d_fake
            optimizerD.step()

            # 更新生成器網絡
            optimizerG.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_images).view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{config.num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {loss_d_total.item():.4f}\tLoss_G: {loss_g.item():.4f}')

            g_losses.append(loss_g.item())
            d_losses.append(loss_d_total.item())

            # 每隔一定迭代次數檢查生成器的進度
            if (iters % 500 == 0) or ((epoch == config.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    show_images(fake[:16], epoch)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return generator, discriminator, g_losses, d_losses, img_list


def main():
    """
    主函數，用於配置和啟動GAN的訓練過程。
    """
    # 加載配置
    config = Config()

    # 創建數據集
    dataset = create_dataset('custom', config.dataroot, config.image_size, config.num_classes)

    # 創建數據加載器
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 初始化生成器和判別器
    generator = Generator(config.ngpu).to(config.device)
    discriminator = Discriminator(config.ngpu).to(config.device)

    # 應用權重初始化
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # 啟動訓練過程
    train(config, dataloader, generator, discriminator)


if __name__ == "__main__":
    main()
