import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
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
    model = EncoderDecoderModel()
    model.load_state_dict(torch.load(model_path))
    return model

def segment_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    print(image[0,:,10,10])
    with torch.no_grad():
        model.eval()
        output = model(image)
    return output.squeeze(0)  # Remove the batch dimension

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model_path = 'encoder_decoder_model.pth'
    model = load_model(model_path)

    image_path = 'test2.png'
    segmented_image = segment_image(model, image_path, transform)

    # Convert tensor image to PIL image and display
    trans = transforms.ToPILImage()
    segmented_image_pil = trans(segmented_image)
    segmented_image_pil.show()

if __name__ == '__main__':
    main()
