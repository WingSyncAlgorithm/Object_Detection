import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
from model import SegNet3, SegNet, SegNet2, DeepSegNet, EncoderDecoderModel



    
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

def segment_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    #print(image[0,:,10,10])
    with torch.no_grad():
        model.eval()
        output = model(image)
    B = torch.argmax(output, dim=1, keepdim=True)
    colors = torch.tensor([
        [0, 0, 0],
        [0, 128, 0],
        [64, 0, 128],
        [64, 64, 0],
        [128, 0, 0],
        [128, 64, 128],
        [128, 128, 0],
        [192, 0, 192]
    ], dtype=torch.float32)

    # 使用張量 B 作為索引來從 colors 張量中選擇顏色
    
    C = colors[B[0, 0]]
    # 轉置結果以獲得正確的形狀 (1, 3, 256, 256)
    output = C.permute(2, 0, 1).unsqueeze(0)
    return output.squeeze(0)  # Remove the batch dimension

def load_model(model_path):
    model = SegNet(3,8)
    #model = DeepSegNet(3,8)
    #model = EncoderDecoderModel()
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    size = 128
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])

    #model_path = 'saved_models\encoder_decoder_model_cat_SegNet3_epoch_72.pth'
    #model_path = 'saved_models\encoder_decoder_model_cat_SegNet2_epoch_60.pth'
    #model_path = 'saved_models\encoder_decoder_model_cat_size512_SegNet3_epoch_198.pth'
    model_path = 'saved_models\encoder_decoder_model_cat_size_SegNet_batch1_epoch_14.pth'
    #model_path = 'saved_models\encoder_decoder_model_cat_block6_6_epoch_10.pth'
    model = load_model(model_path)
    
    save_dir = 'saved_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seq_list = [os.path.join("uavid_v1.5_official_release_image/uavid_test/Seq21/Images", f) for f in os.listdir("uavid_v1.5_official_release_image/uavid_test/Seq21/Images") if f.endswith('.png')]
    
    for image_path in seq_list:
        image_path = 'test.png'
        segmented_image = segment_image(model, image_path, transform)

        # Convert tensor image to PIL image and display
        trans = transforms.ToPILImage()
        #print(segmented_image[:,128,128])
        segmented_image_pil = trans(segmented_image/255.0)
        #print(segmented_image_pil.getpixel((128,128)))
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = save_dir+"/"+filename+"_segmented.png"
        segmented_image_pil.save(save_path)
        break
    segmented_image_pil.show()

if __name__ == '__main__':
    main()
