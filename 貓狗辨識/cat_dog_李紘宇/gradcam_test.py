import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.target_layer.register_forward_hook(self.save_target_layer_output)
        self.gradient = None
    
    def save_target_layer_output(self, module, input, output):
        self.target_layer_output = output
    
    def backward_hook(self, grad):
        print(grad)
        self.gradient = grad.unsqueeze(0)
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x):
        output = self.forward(x)
        output.register_hook(self.backward_hook)
        
        self.model.zero_grad()
        target_class = torch.argmax(output)
        output[0, target_class].backward()

        gradient = self.gradient.cpu().data.numpy()[0]
        target_layer_output = self.target_layer_output.cpu().data.numpy()[0]

        weights = np.mean(gradient, axis=1)
        cam = np.zeros(target_layer_output.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target_layer_output[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, x.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ...（載入模型和圖像）
# 創建模型
# 設置影像大小和類別數量
img_size = 128  # 與訓練時相同的圖片尺寸
num_classes = 2  # 貓和狗兩個類別

# 建立模型並載入訓練好的權重
model = CNNModel(input_shape=(3, img_size, img_size), num_classes=num_classes)

# 載入訓練好的模型權重
model_path = "model_20.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# 載入測試圖像
test_image_path = "test.jpg"  # 請提供測試圖像的路徑
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (img_size, img_size))
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)


# 創建 Grad-CAM 實例，指定目標層
gradcam = GradCAM(model=model, target_layer=model.conv_layers[-1])

# 將圖像轉換為 PyTorch Tensor 格式並進行 Grad-CAM 計算
input_tensor = transforms.ToTensor()(test_image).unsqueeze(0)
gradcam_map = gradcam(input_tensor)

# 將 Grad-CAM 圖像轉換為熱度圖
heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# 將熱度圖的形狀調整為與原始圖像相同
heatmap = cv2.resize(heatmap, test_image.shape[:2][::-1])

# 合併圖像和熱度圖
superimposed_img = heatmap + test_image


# 規範化圖像
superimposed_img_normalized = (superimposed_img - superimposed_img.min()) / (superimposed_img.max() - superimposed_img.min())

# 顯示規範化後的圖像
plt.imshow(superimposed_img_normalized)
plt.axis('off')
plt.show()

# 獲取模型的預測結果
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax().item()

# 判斷是貓還是狗
class_labels = ["貓", "狗"]
predicted_label = class_labels[predicted_class]

print(predicted_label)

