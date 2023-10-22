import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self,in_channels,num_classes=10):
        super(SegNet, self).__init__()
        self.enconv1=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enconv2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enconv3=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enconv4=nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.enconv5=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.deconv1=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.deconv2=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.deconv3=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv4=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv5=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        out=self.enconv1(x)
        out,idx1=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv2(out)
        out,idx2=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv3(out)
        out,idx3=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv4(out)
        out,idx4=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv5(out)
        out,idx5=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=F.max_unpool2d(out,indices=idx5,kernel_size=2,stride=2)
        out=self.deconv1(out)
        out=F.max_unpool2d(out,indices=idx4,kernel_size=2,stride=2)
        out=self.deconv2(out)
        out = F.max_unpool2d(out, indices=idx3, kernel_size=2, stride=2)
        out=self.deconv3(out)
        out = F.max_unpool2d(out, indices=idx2, kernel_size=2, stride=2)
        out=self.deconv4(out)
        out = F.max_unpool2d(out, indices=idx1, kernel_size=2, stride=2)
        out=self.deconv5(out)
        out=self.softmax(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model(model_path):
    model = SegNet(3,3)
    #model = EncoderDecoderModel()
    model.load_state_dict(torch.load(model_path))
    return model

def segment_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    #print(image[0,:,10,10])
    with torch.no_grad():
        model.eval()
        output = model(image)
    return output.squeeze(0)  # Remove the batch dimension

def main():
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    model_path = 'saved_models\encoder_decoder_model_gpu_epoch_41.pth'
    model = load_model(model_path)

    image_path = 'test.png'
    segmented_image = segment_image(model, image_path, transform)

    # Convert tensor image to PIL image and display
    trans = transforms.ToPILImage()
    segmented_image_pil = trans(segmented_image)
    print(segmented_image_pil.getpixel((10,10)))
    segmented_image_pil.show()

if __name__ == '__main__':
    main()
