import numpy as np
import  cv2
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models#直接从官方的torchvision中到入库
from torchvision import transforms
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Batch Normalization after the first convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),  # Batch Normalization after the second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization after the third convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),  # Batch Normalization after the fifth convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 計算全連接層的輸入尺寸
        conv_output_size = input_shape[1] // 32

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size * conv_output_size * 1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            #nn.Softmax(dim=1)
        )

        # 初始化全連接層的權重
        #nn.init.xavier_uniform_(self.fc_layers[1].weight)
        #nn.init.xavier_uniform_(self.fc_layers[3].weight)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform): # 在定義物件時，要輸入模型model(是class CNN的物件)，欲計算gradcam的層target_layers（是一個陣列），reshape_transformg是一個修改陣列形狀的函式
        self.model = model # 設定模型為輸入的model
        self.gradients = [] # 初始化list，用以儲存梯度
        self.activations = [] # 初始化list，用以儲存層的輸出值
        self.reshape_transform = reshape_transform # 設定修改陣列形狀的函式
        self.handles = [] # 設置一個list，用以儲存每次註冊鉤子函數的動作
        for target_layer in target_layers: # 跑遍欲處理的層
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation)) # 將一個前向鉤子註冊到target_layer上，這個鉤子將在前向傳播過程中被觸發，並調用self.save_activation函式，保存輸出值
                                           # 將註冊動作添加進handles
            # Backward compatibility with older pytorch versions:
            #这里的if语句主要处理的是pytorch版本兼容问题
            if hasattr(target_layer, 'register_full_backward_hook'): # 檢查target_layer物件中，有沒有register_full_backward_hook函式，判別pytorch版本
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient)) # 註冊反向鉤子，反向傳播時，調用self.save_gradient函式，保存梯度
                                             # 將註冊動作添加進handles
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient)) # 註冊反向鉤子，反向傳播時，調用self.save_gradient函式，保存梯度
                                             # 將註冊動作添加進handles
#正向傳播
    def save_activation(self, module, input, output):
        activation = output #獲取模型輸出值
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation) # 修改activation形狀
        self.activations.append(activation.cpu().detach()) # 輸出值轉移到 CPU 上，並且將其添加到self.activations列表中
#反向传播有高层流向低层【保存的方式相反】
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0] # 取target_layer輸出的梯度
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad) # 修改grad形狀
        self.gradients = [grad.cpu().detach()] + self.gradients # 將梯度移到 CPU 上並分離出來，然後加入到gradients列表中
    #正向传播
    def __call__(self, x): # 輸入圖片，輸出個類別機率
        self.gradients = []
        self.activations = []
        return self.model(x) # 正向傳播,輸出輸出層的值

    def release(self):
        for handle in self.handles: # 遍歷所有鉤子
            handle.remove() # 將註冊的鉤子全部清除


class GradCAM:
    def __init__(self,
                 model, # 輸入模型model(是class CNN的物件)
                 target_layers, # 欲計算gradcam的層target_layers（是一個陣列）
                 reshape_transform=None, # 輸入調整陣列形狀的函式
                 # 使用不同的设备，默认使用gpu
                 use_cuda=True):
        self.model = model.eval() # 模型设置为验证模式
        self.target_layers = target_layers # 設置欲處理的層
        self.reshape_transform = reshape_transform # 設定修改陣列形狀的函式
        self.cuda = use_cuda # 是否使用gpu，True or False
        if self.cuda: # 如果是True
            self.model = model.cuda() # 使用gpu
        #ActivationsAndGradients正向传播获得的A和反向传播获得的A'
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform) # 定義ActivationsAndGradients的物件

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads): #輸入梯度grads[batch,通道,高,寬]
        return np.mean(grads, axis=(2, 3), keepdims=True) # 輸出特徵圖的每一點梯度的平均，並保持矩陣形狀為[batch,通道,1,1]

    @staticmethod
    def get_loss(output, target_category):
        """
        將預測圖作为loss回传

        :param output: 模型的輸出。
        :param target_category: 目標類別的索引。
        :return: 損失值。
        """
        loss = 0  # 初始化損失
        for i in range(len(target_category)):
            print(output)
            loss = loss + output[i, target_category[i]] # output[i, target_category[i]]表示輸出層第i類的輸出值
        return loss

    def get_cam_image(self, activations, grads):
        """
        獲取 CAM 圖像。

        :param activations: 特徵圖。
        :param grads: 梯度值。
        :return: CAM 圖像。
        """
        weights = self.get_cam_weights(grads)  # 獲取 CAM 權重
        weighted_activations = weights * activations  # 權重乘以特徵圖
        cam = weighted_activations.sum(axis=1)  # 對通道進行總和，獲得 CAM 圖像

        return cam


    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)  # 取得輸入張量的寬度和高度
        return width, height  # 回傳寬度和高度


    def compute_cam_per_layer(self, input_tensor):
        """
        計算每層的 CAM。

        :param input_tensor: 輸入張量。
        :return: 一個列表，包含每層的 CAM。
        """
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]  # 取得每層的激活值
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]  # 取得每層的梯度值
        target_size = self.get_target_width_height(input_tensor)  # 取得圖片張量尺寸

        cam_per_target_layer = []  # 用於儲存每層的 CAM

        
        for layer_activations, layer_grads in zip(activations_list, grads_list): # 循環遍歷每層的激活值和梯度值
            cam = self.get_cam_image(layer_activations, layer_grads)  # 計算 CAM
            cam[cam < 0] = 0  # 將所有小於零的值變為零（相當於使用了 ReLU 激活函數）
            scaled = self.scale_cam_image(cam, target_size)  # 尺寸縮放 CAM
            cam_per_target_layer.append(scaled[:, None, :])  # 加入列表中

        return cam_per_target_layer  # 返回每層的 CAM


    def aggregate_multi_layers(self, cam_per_target_layer):
        """
        聚合多層 CAM 並對結果進行後處理。

        :param cam_per_target_layer: 一個列表，包含了多層的 CAM。
        :return: 經過聚合並後處理後的 CAM。
        """
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)  # 將多層 CAM 進行水平連接
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)  # 取最大值，並消除負值
        result = np.mean(cam_per_target_layer, axis=1)  # 對每個目標進行平均
        return self.scale_cam_image(result)  # 對聚合後的 CAM 進行尺寸縮放


    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []  # 創建一個空的列表 result 用來儲存處理後的影像
        for img in cam:  # 進入迴圈，對於 cam 中的每張影像執行以下操作
            img = img - np.min(img)  # 將影像中的最小值減去影像中的最小值，以進行正規化處理
            img = img / (1e-7 + np.max(img))  # 將影像除以影像中的最大值，同樣為了正規化處理
            if target_size is not None:  # 檢查目標尺寸是否為空值
                img = cv2.resize(img, target_size)  # 若有指定目標尺寸，使用 OpenCV (cv2) 進行影像縮放處理
            result.append(img)  # 將處理後的影像加入 result 列表
        result = np.float32(result)  # 將 result 轉換成浮點數型態的 NumPy 陣列
        return result  # 回傳處理後的影像陣列


    def __call__(self, input_tensor, target_category=None):
        name = ["Crabs", "Dolphin", "blue_ringed_octopus", "Sea Urchins", "Seahorse", "Turtle_Tortoise", "Jelly Fish", "Lobster", "Nudibranchs", "Seal"]

        if self.cuda: # 判斷是否使用gpu
            input_tensor = input_tensor.cuda() # 將PyTorch 張量 input_tensor 移動到 GPU 上

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor) # 將圖片輸入self.activations_and_grads()，執行class ActivationsAndGradients中的__call__，輸出神經網路輸出層的輸出值
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
            print(target_category)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1) 
            print(target_category)
            
            '''
            output: 這是模型的輸出，通常是一個 PyTorch 張量
            .cpu(): 將張量移到 CPU 上進行後續操作（如果之前在 GPU 上進行了計算）。
            .data: 獲取張量的數據部分，不包括梯度信息。
            .numpy(): 將數據轉換為 NumPy 陣列。
            np.argmax(..., axis=-1): 在最後一個維度上進行最大值的尋找，輸出索引。
            '''
            print(f"category id: {target_category} {name[target_category[0]]}") # 輸出最可能的類別
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad() # 清空历史梯度信息
        loss = self.get_loss(output, target_category) # 獲取輸出層的輸出值，當作loss
        loss.backward(retain_graph=True) # 把輸出值當loss作反向傳播

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor) # 獲取多層的gradcam
        return self.aggregate_multi_layers(cam_per_layer) # 把多層gradcam做平均

    def __del__(self): # 類別實例被銷毀時執行的方法。釋放相關資源
        self.activations_and_grads.release()

    def __enter__(self): # 進入 with block 時執行的方法。返回實例本身
        return self

    def __exit__(self, exc_type, exc_value, exc_tb): # 離開 with block 時執行的方法，釋放相關資源
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError): # 檢查變數 exc_value 是否屬於 IndexError 類型
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)  # 將 CAM 遮罩應用成熱度圖
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 若使用 RGB 格式，轉換顏色通道
    heatmap = np.float32(heatmap) / 255  # 正規化熱度圖的值到範圍 [0, 1]

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")  # 檢查輸入圖像的值範圍，必須在0~1之間

    cam = heatmap + img  # 將熱度圖和原始圖像相加
    cam = cam / np.max(cam)  # 正規化疊加後的圖像的值到範圍 [0, 1]
    return np.uint8(255 * cam)  # 將圖像的值範圍轉換回 [0, 255]


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape  # 取得輸入圖像的高、寬和通道數

    if w == h == size:  # 若圖像已經是目標尺寸，則直接返回原圖
        return img

    if w < h:  # 若寬度小於高度
        ratio = size / w  # 計算裁剪比例
        new_w = size  # 新的寬度為目標尺寸
        new_h = int(h * ratio)  # 新的高度為經過比例縮放後的高度
    else:  # 若高度小於寬度
        ratio = size / h  # 計算裁剪比例
        new_h = size  # 新的高度為目標尺寸
        new_w = int(w * ratio)  # 新的寬度為經過比例縮放後的寬度

    img = cv2.resize(img, dsize=(new_w, new_h))  # 將圖像縮放至新的尺寸

    if new_w == size:  # 若新的寬度等於目標尺寸
        h = (new_h - size) // 2  # 計算上下裁剪的邊緣大小
        img = img[h: h+size]  # 進行上下裁剪
    else:  # 若新的高度等於目標尺寸
        w = (new_w - size) // 2  # 計算左右裁剪的邊緣大小
        img = img[:, w: w+size]  # 進行左右裁剪

    return img  # 返回裁剪後的圖像


def apply_gradcam_to_directory(directory_path, model, target_layers, img_size=224):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    # Class names for prediction
    class_names = ["Crabs", "Dolphin", "blue_ringed_octopus", "Sea Urchins", "Seahorse", "Turtle_Tortoise", "Jelly Fish", "Lobster", "Nudibranchs", "Seal"]

    # Check if directory exists
    if not os.path.exists(directory_path):
        print("Directory does not exist!")
        return
    
    # Loop through each image in the directory
    for img_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_name)
        
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Skip images with filenames containing "_cam_"
        if "_cam_" in img_name:
            continue
        
        test_image = cv2.imread(img_path)
        test_image = cv2.resize(test_image, (img_size, img_size))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        input_tensor = transforms.ToTensor()(test_image).unsqueeze(0)
        
        output = model(input_tensor)
        predicted_class = class_names[torch.argmax(output).item()]
        
        target_category = None
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(test_image / 255., grayscale_cam, use_rgb=True)
        
        # Save or display the visualization
        output_filename = os.path.splitext(img_name)[0] + "_cam_" + predicted_class + os.path.splitext(img_name)[1]
        output_path = os.path.join(directory_path, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved GradCAM result for {img_name} to {output_path}")

def main():
    directory_path = "test"  # Set to your images directory path
    img_size = 224
    num_classes = 6
    model = CNNModel(input_shape=(3, img_size, img_size), num_classes=num_classes)
    model_path = "model_kaggle_6class_processed_epoch50_batchnorm.pth"
    model.load_state_dict(torch.load(model_path))
    target_layers = [model.conv_layers]
    
    apply_gradcam_to_directory(directory_path, model, target_layers)
    
main()