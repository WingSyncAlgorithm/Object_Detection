import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
from model import SegNet3, SegNet, SegNet2, DeepSegNet, EncoderDecoderModel
import cv2
import numpy as np
import time

# Check if CUDA GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def segment_image(model, image_input, transform):
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Invalid input type for image. Expected a file path or PIL Image object.")

    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add a batch dimension and move to device

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
    ], dtype=torch.float32).to(device)

    C = colors[B[0, 0]]
    output = C.permute(2, 0, 1).unsqueeze(0)
    return output.squeeze(0)  # Remove the batch dimension

def load_model(model_path):
    model = SegNet3(3,8)
    #model = DeepSegNet(3,8)
    #model = EncoderDecoderModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to the device
    return model

def segment_video(model,size, video_path, transform):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_test2_segmented.avi', fourcc, 20.0, (size,size))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            segmented_image = segment_image(model, frame_pil, transform)
            
            # Convert tensor image to PIL image
            trans = transforms.ToPILImage()
            segmented_image_pil = trans(segmented_image.cpu()/255.0)  # Move tensor to CPU for PIL conversion
            
            # Convert the PIL Image back to OpenCV format and write to video
            segmented_image_cv2 = cv2.cvtColor(np.array(segmented_image_pil), cv2.COLOR_RGB2BGR)
            out.write(segmented_image_cv2)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    start_time = time.time()
    size = 512
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])

    #model_path = 'saved_models\encoder_decoder_model_cat_SegNet2_epoch_70.pth'
    model_path = 'saved_models\encoder_decoder_model_cat_size512_SegNet3_epoch_199.pth'
    #model_path = 'saved_models\encoder_decoder_model_cat_epoch_100.pth'
    model = load_model(model_path)
    
    video_path = 'video_test2.mp4'
    segment_video(model, size, video_path, transform)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
