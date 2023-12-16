# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    sqrt_n = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrt_n, sqrt_n, index+1)
        plt.imshow(image.reshape(28, 28))


# Discriminator Loss => BCELoss
def dis_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)


def gen_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)


# GPU
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('GPU State:', device)

# Model
generator = Generator().to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)

# Settings
epochs = 30
learning_rate = 0.0002
batch_size = 64
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# Transform
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])


# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# Train
for epoch in tqdm(range(epochs),desc='epoch'):
    epoch += 1

    for times, data in tqdm(enumerate(train_loader)):
        times += 1
        real_inputs = data[0].to(device)
        test = 255 * (0.5 * real_inputs[0] + 0.5)

        real_inputs = real_inputs.view(-1, 784)
        real_outputs = discriminator(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = generator(noise)
        fake_outputs = discriminator(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        # Zero the parameter gradients
        dis_optimizer.zero_grad()

        # Backward propagation
        dis_loss = dis_loss_function(outputs, targets)
        dis_loss.backward()
        dis_optimizer.step()

        # Generator
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_inputs = generator(noise)
        fake_outputs = discriminator(fake_inputs)

        gen_loss = gen_loss_function(fake_outputs)
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        if times % 100 == 0 or times == len(train_loader):
            print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader), dis_loss.item(), g_loss.item()))

    imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
    show_images(imgs_numpy[:16])
    plt.show()

    if epoch % 10 == 0:
        torch.save(generator, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')


print('Training Finished.')
print('Cost Time: {}s'.format(time.time()-start_time))
