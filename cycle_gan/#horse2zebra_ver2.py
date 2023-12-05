#horse2zebra_ver2

import numpy as np
import pandas as pd
import os, math, sys
import time, datetime
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)
import warnings
warnings.filterwarnings("ignore")

# epoch to start training from
epoch_start = 25
# number of epochs of training
n_epochs = 26
# name of the dataset
dataset_path = r"C:\Users\jacky\OneDrive\文件\cyclegan\horse2zebra"
# size of the batches"
batch_size = 4
# adam: learning rate
lr = 0.0001
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of first order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 1
# number of cpu threads to use during batch generation
n_workers = 8
# size of image height
img_height = 256
# size of image width
img_width = 256
# number of image channels
channels = 3
# interval between saving generator outputs
sample_interval = 100
# interval between saving model checkpoints
checkpoint_interval = -1
# number of residual blocks in generator
n_residual_blocks = 9
# cycle loss weight
lambda_cyc = 10.0
# identity loss weight
lambda_id = 5.0
# Development / Debug Mode
debug_mode = False

# Create images and checkpoint directories
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def to_rgb(image):
    """將非RGB格式的圖片轉換為RGB格式

    Args:
        image (PIL.Image): 非RGB格式的圖片物件
        
    Returns:
        PIL.Image: RGB格式的圖片物件
        
    """
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ReplayBuffer:
    def __init__(self, max_size=50):
        """buffer的初始化

        Args:
            max_size(int):buffer的最大值，預設=50
            
        """
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """將提供的數據加入緩衝區，同時50%彈出舊的數據

    
        Args:
            data (torch.Tensor):包含數據的tensor

        Returns:
            torch.Tensor:經過處理的tensor
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.tensor(torch.cat(to_return))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))
        if debug_mode:
            self.files_A = self.files_A[:100]
            self.files_B = self.files_B[:100]

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


# Image transformations
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
train_dataloader = DataLoader(
    ImageDataset(f"{dataset_path}", transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
# Test data loader
test_dataloader = DataLoader(
    ImageDataset(f"{dataset_path}", transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

def weights_init_normal(m):
    """初始化模型權重的函數
    卷積層使用正態分佈初始化
    標準化層使用正態分佈初始化權重和常數初始化偏差。

    Args:
        m (nn.Module): 要初始化權重的模型層
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """
        定義殘差塊

        Args:
            in_features(int): 輸入特徵的channel
        """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        """前向傳播
            Args:
                x(tensor):輸入
            Returns:
                x+self.block(x) (tensor):輸出
        """
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
        定義生成器的ResNet模型。

        Args:
            input_shape (tuple): 輸入圖像的形狀 (通道數, 高度, 寬度)
            num_residual_blocks (int): 殘差塊數量
        """
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """前向傳播
        Args:
            x (torch.Tensor):輸入

        Returns:
            self.model(x)(torch.Tensor):輸出
        
        """
        return self.model(x)


class Discriminator(nn.Module):
    """
        定義鑑別器

        Args:
            input_shape (tuple): 輸入圖像的形狀 (通道數, 高度, 寬度)
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """
            構建鑑別器的基本block，為了下面方便建構，以下都是下採樣。

            Args:
                in_filters (int): 輸入通道數
                out_filters (int): 輸出通道數
                normalize (bool): 是否進行正歸一化，預設為True

            Returns:
                layer(list): 包含下採樣層組成的列表
            """
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (channels, img_height, img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# Initialize weights
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

train_counter = []
train_losses_gen, train_losses_id, train_losses_gan, train_losses_cyc = [], [], [], []
train_losses_disc, train_losses_disc_a, train_losses_disc_b = [], [], []

test_counter = [2*idx*len(train_dataloader.dataset) for idx in range(epoch_start+1, n_epochs+1)]
test_losses_gen, test_losses_disc = [], []


def train_and_test():
    """
    進行訓練及測試
    並把loss最低的權重紀錄起來
    """
    for epoch in range(epoch_start, n_epochs):
        
        #### Training
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_a = loss_disc_b = 0.0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
        for batch_idx, batch in enumerate(tqdm_bar):
            print("1\n")
            # Set model input
            real_A = torch.tensor(batch["A"].type(Tensor))
            real_B = torch.tensor(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = torch.tensor(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = torch.tensor(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            ### Train Generators
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = lambda_id * loss_identity + loss_GAN + lambda_cyc * loss_cycle
            loss_G.backward()
            optimizer_G.step()

            ### Train Discriminator-A
            D_A.train()
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            ### Train Discriminator-B
            D_B.train()
            optimizer_D_B.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            loss_D = (loss_D_A + loss_D_B) / 2

            ### Log Progress
            loss_gen += loss_G.item(); loss_id += loss_identity.item(); loss_gan += loss_GAN.item(); loss_cyc += loss_cycle.item()
            loss_disc += loss_D.item(); loss_disc_a += loss_D_A.item(); loss_disc_b += loss_D_B.item()
            train_counter.append(2*(batch_idx*batch_size + real_A.size(0) + epoch*len(train_dataloader.dataset)))
            train_losses_gen.append(loss_G.item()); train_losses_id.append(loss_identity.item()); train_losses_gan.append(loss_GAN.item()); train_losses_cyc.append(loss_cycle.item())
            train_losses_disc.append(loss_D.item()); train_losses_disc_a.append(loss_D_A.item()); train_losses_disc_b.append(loss_D_B.item())
            tqdm_bar.set_postfix(Gen_loss=loss_gen/(batch_idx+1), identity=loss_id/(batch_idx+1), adv=loss_gan/(batch_idx+1), cycle=loss_cyc/(batch_idx+1),
                                Disc_loss=loss_disc/(batch_idx+1), disc_a=loss_disc_a/(batch_idx+1), disc_b=loss_disc_b/(batch_idx+1))

        #### Testing
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_a = loss_disc_b = 0.0
        tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
        for batch_idx, batch in enumerate(tqdm_bar):

            # Set model input
            real_A = torch.tensor(batch["A"].type(Tensor))
            real_B = torch.tensor(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = torch.tensor(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = torch.tensor(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            ### Test Generators
            G_AB.eval()
            G_BA.eval()
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            ### Test Discriminator-A
            D_A.eval()
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            ### Test Discriminator-B
            D_B.eval()
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D = (loss_D_A + loss_D_B) / 2
            
            ### Log Progress
            loss_gen += loss_G.item(); loss_id += loss_identity.item(); loss_gan += loss_GAN.item(); loss_cyc += loss_cycle.item()
            loss_disc += loss_D.item(); loss_disc_a += loss_D_A.item(); loss_disc_b += loss_D_B.item()
            tqdm_bar.set_postfix(Gen_loss=loss_gen/(batch_idx+1), identity=loss_id/(batch_idx+1), adv=loss_gan/(batch_idx+1), cycle=loss_cyc/(batch_idx+1),
                                Disc_loss=loss_disc/(batch_idx+1), disc_a=loss_disc_a/(batch_idx+1), disc_b=loss_disc_b/(batch_idx+1))
            
            # If at sample interval save image
            if random.uniform(0,1)<0.4:
                # Arrange images along x-axis
                real_A = make_grid(real_A, nrow=1, normalize=True)
                real_B = make_grid(real_B, nrow=1, normalize=True)
                fake_A = make_grid(fake_A, nrow=1, normalize=True)
                fake_B = make_grid(fake_B, nrow=1, normalize=True)
                # Arange images along y-axis
                image_grid = torch.cat((real_A, fake_B, real_B, fake_A), -1)
                save_image(image_grid, f"images/{batch_idx}.png", normalize=False)

        test_losses_gen.append(loss_gen/len(test_dataloader))
        test_losses_disc.append(loss_disc/len(test_dataloader))

        # Save model checkpoints
        if np.argmin(test_losses_gen) == len(test_losses_gen)-1:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/G_AB.pth")
            torch.save(G_BA.state_dict(), "saved_models/G_BA.pth")
            torch.save(D_A.state_dict(), "saved_models/D_A.pth")
            torch.save(D_B.state_dict(), "saved_models/D_B.pth")

train_and_test()

def plot_generator_losses(train_counter, train_losses_gen, train_losses_id, train_losses_gan, train_losses_cyc,
                           test_counter, test_losses_gen):
    """
    將提供的生成器損失數據繪製成圖表，並存成存為 HTML 文件。

    Args:
        train_counter (list): 訓練步數的列表。
        train_losses_gen (list): 訓練生成器總體損失的列表。
        train_losses_id (list): 訓練生成器身份損失的列表。
        train_losses_gan (list): 訓練生成器GAN損失的列表。
        train_losses_cyc (list): 訓練生成器循環損失的列表。
        test_counter (list): 測試步數的列表。
        test_losses_gen (list): 測試生成器總體損失的列表。

    Returns:
        None
    """
    fig = go.Figure()

    # 添加訓練數據的線
    fig.add_trace(go.Scatter(x=train_counter, y=train_losses_gen, mode='lines', name='訓練總體損失 (Loss_G)'))
    fig.add_trace(go.Scatter(x=train_counter, y=train_losses_id, mode='lines', name='訓練身份損失'))
    fig.add_trace(go.Scatter(x=train_counter, y=train_losses_gan, mode='lines', name='訓練GAN損失'))
    fig.add_trace(go.Scatter(x=train_counter, y=train_losses_cyc, mode='lines', name='訓練循環損失'))

    # 添加測試數據的標記
    fig.add_trace(go.Scatter(x=test_counter, y=test_losses_gen, marker_symbol='star-diamond',
                             marker_color='orange', marker_line_width=1, marker_size=9, mode='markers',
                             name='Test Disc Loss (Loss_G)'))

    # 更新圖表佈局
    fig.update_layout(
        width=1000,
        height=500,
        title="Train vs. Test Discriminator Loss",
        xaxis_title="Number of training examples seen (A+B)",
        yaxis_title="Discriminator Losses"),

    # 顯示圖表
    fig.show()

plot_generator_losses(train_counter, train_losses_gen, train_losses_id, train_losses_gan, train_losses_cyc,
                       test_counter, test_losses_gen)

def visualize_horse_to_zebra(G_AB, test_dataloader, num_samples):
    """
    將提供的生成器模型用於將馬的圖片轉換為斑馬

    Args:
        G_AB (torch.nn.Module): 把馬圖片變為斑馬模型
        test_dataloader (DataLoader): 測試集的 DataLoader
        num_samples (int): 要幾張

    """
    # 根據 GPU 可用性設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 將模型移動到指定的裝置上
    G_AB = G_AB.to(device)

    # 從 DataLoader 中獲取測試樣本
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))

    for batch_idx, batch in enumerate(test_dataloader):
        if batch_idx >= num_samples:
            break

        input_horse = batch["A"].type(Tensor).to(device)

        # 使用生成器模型進行轉換
        generated_zebra = G_AB(input_horse)

        # 將原始和生成的圖片轉換為 NumPy 數組
        original_img = make_grid(input_horse.cpu(), nrow=1, normalize=True).permute(1, 2, 0).detach().numpy()
        generated_img = make_grid(generated_zebra.cpu(), nrow=1, normalize=True).permute(1, 2, 0).detach().numpy()

        # 顯示原始和生成的圖片
        axes[batch_idx, 0].imshow(original_img)
        axes[batch_idx, 0].axis("off")
        axes[batch_idx, 0].set_title(f"before {batch_idx + 1}")

        axes[batch_idx, 1].imshow(generated_img)
        axes[batch_idx, 1].axis("off")
        axes[batch_idx, 1].set_title(f"after {batch_idx + 1}")

    plt.tight_layout()
    plt.show()

visualize_horse_to_zebra(G_AB,test_dataloader,num_samples=100)