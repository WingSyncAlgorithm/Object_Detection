import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 資料預處理與加載
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# CNN 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 攤平特徵
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 訓練與評估模型
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
for epoch in range(10):
    net.train()  # 設定為訓練模式
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    train_loss = running_loss / len(trainloader)
    
    net.eval()  # 設定為評估模式
    correct = 0
    total = 0
    val_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
            
    val_acc = 100. * correct / total
    val_loss /= len(testloader)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 繪製 learning curve 和 loss curve
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend(loc='best')
plt.show()

# 繪製 confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Grad-CAM 部分
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []
        self.model.eval()

    def forward_hook(self, module, input, output):
        self.feature_maps.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def register_hooks(self):
        self.feature_maps = []
        self.gradients = []
        for module in self.model.children():
            module.register_forward_hook(self.forward_hook)
            module.register_backward_hook(self.backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        self.model.eval()
        pred = self.model(input_image)
        target = torch.tensor([target_class], dtype=torch.long).to(input_image.device)

        loss = F.cross_entropy(pred, target)
        loss.backward(retain_graph=True)

        gradients = self.gradients[0].to(input_image.device)
        
        # 確保 gradients 張量至少有3個維度
        if gradients.dim() < 3:
            gradients = gradients.unsqueeze(2).unsqueeze(3)  # 添加兩個維度

        pooled_gradients = F.adaptive_avg_pool2d(gradients, (1, 1))
        
        # 移除多餘的維度
        pooled_gradients = pooled_gradients[0, :, :, :]

        feature_maps = self.feature_maps[0].to(input_image.device)
        for i in range(gradients.size(1)):
            feature_maps[:, i, :, :] *= pooled_gradients[i, :, :]

        heatmap = torch.mean(feature_maps, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        cam = F.interpolate(heatmap, size=(input_image.size()[-2], input_image.size()[-1]), mode="bilinear", align_corners=False)
        cam = cam[0, 0, :, :]

        return cam.detach().cpu().numpy()

# 初始化設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 隨機選取一張測試圖片，顯示其原圖和 Grad-CAM 結果
gradcam = GradCAM(model=net)
gradcam.register_hooks()

random_index = np.random.choice(len(testset))
test_image, test_label = testset[random_index]

# 轉換成模型輸入所需的格式
test_image = test_image.unsqueeze(0).to(device)

# 確保模型處於評估模式，以避免 dropout 等層的影響
net.eval()

# 取得模型預測的類別
with torch.no_grad():
    predicted_class = net(test_image).argmax().item()

# 生成 Grad-CAM
heatmap = gradcam.generate_cam(test_image, target_class=predicted_class)

# 將 heatmap 轉為 OpenCV 格式
heatmap = cv2.resize(heatmap, (test_image.size()[-1], test_image.size()[-2]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 將 heatmap 與原圖結合
original_image = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
superimposed_image = heatmap * 0.7 + original_image * 0.3
superimposed_image /= superimposed_image.max()

# 顯示圖像和 Grad-CAM 結果
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(original_image, cmap='gray')
plt.title(f"True Class: {test_label}")
plt.axis('off')
plt.subplot(122)
plt.imshow(superimposed_image)
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()

