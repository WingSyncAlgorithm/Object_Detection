import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import cv2
import numpy as np

class SegNet3(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SegNet3, self).__init__()
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
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.enconv6=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.enconv7=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.enconv8=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv_2=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv_1=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv0=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv1=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
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
        out=self.enconv6(out)
        out,idx6=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv7(out)
        out,idx7=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv8(out)
        out,idx8=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        
        out=F.max_unpool2d(out,indices=idx8,kernel_size=2,stride=2)
        out=self.deconv_2(out)
        out=F.max_unpool2d(out,indices=idx7,kernel_size=2,stride=2)
        out=self.deconv_1(out)
        out=F.max_unpool2d(out,indices=idx6,kernel_size=2,stride=2)
        out=self.deconv0(out)
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
    
class DeepSegNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(DeepSegNet, self).__init__()
        
        # Encoder Blocks
        self.encoders = nn.ModuleList([
            self.encoder_block(in_channels, 64),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256),
            self.encoder_block(256, 256),
            self.encoder_block(256, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512)
        ])
        
        # Decoder Blocks
        self.decoders = nn.ModuleList([
            self.decoder_block(512, 512),
            self.decoder_block(512, 512),
            self.decoder_block(512, 512),
            self.decoder_block(512, 512),
            self.decoder_block(512, 256),
            self.decoder_block(256, 256),
            self.decoder_block(256, 128),
            self.decoder_block(128, 64),
            self.decoder_block(64, 64),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            )
        ])
        
        self.softmax = nn.Softmax(dim=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x, _ = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        for i, decoder in enumerate(self.decoders):
            if i < 9:  # Exclude the last layer from unpooling
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
                x = x + skips[-(i+1)]  # Add skip connection
            x = decoder(x)
        
        return self.softmax(x)
    
class SegNet(nn.Module):
    def __init__(self,in_channels,num_classes):
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

class SegNet2(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SegNet2, self).__init__()
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
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.enconv6=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv0=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv1=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
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
        out=self.enconv6(out)
        out,idx6=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=F.max_unpool2d(out,indices=idx6,kernel_size=2,stride=2)
        out=self.deconv0(out)
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
    
class SegNet4(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SegNet4, self).__init__()
        self.enconv1=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=5,padding=2),
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
        self.enconv4_5=nn.Sequential(
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,kernel_size=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enconv5=nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.enconv6=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv0=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.deconv1=nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv1_5=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=1,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
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
        out=self.enconv4_5(out)
        out,idx4_5=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv5(out)
        out,idx5=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        out=self.enconv6(out)
        out,idx6=F.max_pool2d(out,kernel_size=2,stride=2,return_indices=True)
        
        out=F.max_unpool2d(out,indices=idx6,kernel_size=2,stride=2)
        out=self.deconv0(out)
        out=F.max_unpool2d(out,indices=idx5,kernel_size=2,stride=2)
        out=self.deconv1(out)
        out=F.max_unpool2d(out,indices=idx4_5,kernel_size=2,stride=2)
        out=self.deconv1_5(out)
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