import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.utils as vutils
import torchvision.datasets as dset
import cv2
import random

# Root directory for dataset
dataroot = "dataset\data"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class CustomDataset2(Dataset): # 繼承Dataset
    def __init__(self, data_dir, img_size, num_classes): # 定義物件時，會執行的初始化函式
        self.data_dir = data_dir # 設置訓練資料的位置
        self.img_size = img_size # 設置圖片大小
        self.num_classes = num_classes # 設置類的數量
        self.categories = ["face"] # 訓練資料放置的資料夾名稱
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
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator3(nn.Module):
    def __init__(self,ngpu):
        super(Generator3, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
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
        #print(self.main(input).size())
        return self.main(input)

class Discriminator2(nn.Module):
    def __init__(self, ngpu, input_shape = (3,image_size,image_size), num_classes = 1):
        super(Discriminator2, self).__init__()
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
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        #print(x.size())
        return x
        
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def show_images(images, epoch):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    print(sqrtn)
    plt.figure(figsize=(10,10))  # 可以根据需要调整大小
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
    
    
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset2(dataroot, image_size, 1)
#dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

#dataset = dset.ImageFolder(root=dataroot,transform=transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator3(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator2(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(16, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    #print("0")
    # For each batch in the dataloader
    for i, data in enumerate(dataloader,0):
        #print("00")
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        #print("1")
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        #print("2")
        output = netD(real_cpu).view(-1)
        #print("2.5")
        
        # Calculate loss on all-real batch
        #print(output.size(),label.size())
        errD_real = criterion(output, label)
        #print("2.55")
        # Calculate gradients for D in backward pass
        errD_real.backward()
        #print("2.6")
        D_x = output.mean().item()
        #print("3")

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #print("4")
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        #print("5")

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        #print("6")

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                show_images(fake[:16],epoch)
                plt.show()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1