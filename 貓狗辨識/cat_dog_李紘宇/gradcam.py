import numpy as np
import cv2
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models  # 直接从官方的torchvision中到入库
from torchvision import transforms
import torch.nn as nn


class CNNModel(nn.Module):  # 定義class CNNModel，並繼承torch.nn.Module
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # 3通道(rgb)輸入,輸出32個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # 負值變0
            nn.MaxPool2d(2),  # 2*2方格取最大值

            # 32通道輸入上一層輸出的32張圖,輸出64個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # 負值變0
            nn.MaxPool2d(2),  # 2*2方格取最大值

            # 64通道輸入上一層輸出的64張圖,輸出128個特徵圖,使用3*3kernel,輸出的圖周圍補一格0
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # 負值變0
            nn.MaxPool2d(2)  # 2*2方格取最大值
        )

        # 計算全連接層的輸入尺寸
        # 在三次池化後的影像尺寸，因為做了三次2*2的MaxPooling
        conv_output_size = input_shape[1] // 2 // 2 // 2
        self.fc_layers = nn.Sequential(  # 定義全連接層
            nn.Flatten(),  # 把所有特徵圖，降成一維張量
            nn.Linear(conv_output_size * conv_output_size * 128,
                      64),  # 接一層神經元，輸入通道數=128張特徵圖的總像素量，輸出64個通道
            nn.ReLU(),  # 接讓負值變0的神經元，輸出與輸入輸量相同
            nn.Linear(64, num_classes),  # 這是輸出層，將上一層輸出的64個輸出值，各類別可能性大小
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 正向傳播經過卷積層
        x = self.fc_layers(x)  # 正向傳播經過全連接層
        return x  # 輸出輸出層的值


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    # 在定義物件時，要輸入模型model(是class CNN的物件)，欲計算gradcam的層target_layers（是一個陣列），reshape_transformg是一個修改陣列形狀的函式
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model  # 設定模型為輸入的model
        self.gradients = []  # 初始化list，用以儲存梯度
        self.activations = []  # 初始化list，用以儲存層的輸出值
        self.reshape_transform = reshape_transform  # 設定修改陣列形狀的函式
        self.handles = []  # 設置一個list，用以儲存每次註冊鉤子函數的動作
        for target_layer in target_layers:  # 跑遍欲處理的層
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))  # 將一個前向鉤子註冊到target_layer上，這個鉤子將在前向傳播過程中被觸發，並調用self.save_activation函式，保存輸出值
            # 將註冊動作添加進handles
            # Backward compatibility with older pytorch versions:
            # 这里的if语句主要处理的是pytorch版本兼容问题
            # 檢查target_layer物件中，有沒有register_full_backward_hook函式，判別pytorch版本
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))  # 註冊反向鉤子，反向傳播時，調用self.save_gradient函式，保存梯度
                # 將註冊動作添加進handles
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))  # 註冊反向鉤子，反向傳播時，調用self.save_gradient函式，保存梯度
                # 將註冊動作添加進handles
# 正向傳播

    def save_activation(self, module, input, output):
        activation = output  # 獲取模型輸出值
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)  # 修改activation形狀
        # 輸出值轉移到 CPU 上，並且將其添加到self.activations列表中
        self.activations.append(activation.cpu().detach())
# 反向传播有高层流向低层【保存的方式相反】

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]  # 取target_layer輸出的梯度
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)  # 修改grad形狀
        # 將梯度移到 CPU 上並分離出來，然後加入到gradients列表中
        self.gradients = [grad.cpu().detach()] + self.gradients
    # 正向传播

    def __call__(self, x):  # 輸入圖片，輸出個類別機率
        self.gradients = []
        self.activations = []
        return self.model(x)  # 正向傳播,輸出輸出層的值

    def release(self):
        for handle in self.handles:  # 遍歷所有鉤子
            handle.remove()  # 將註冊的鉤子全部清除


class GradCAM:
    def __init__(self,
                 model,  # 輸入模型model(是class CNN的物件)
                 target_layers,  # 欲計算gradcam的層target_layers（是一個陣列）
                 reshape_transform=None,  # 輸入調整陣列形狀的函式
                 # 使用不同的设备，默认使用gpu
                 use_cuda=True):
        self.model = model.eval()  # 模型设置为验证模式
        self.target_layers = target_layers  # 設置欲處理的層
        self.reshape_transform = reshape_transform  # 設定修改陣列形狀的函式
        self.cuda = use_cuda  # 是否使用gpu，True or False
        if self.cuda:  # 如果是True
            self.model = model.cuda()  # 使用gpu
        # ActivationsAndGradients正向传播获得的A和反向传播获得的A'
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)  # 定義ActivationsAndGradients的物件

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):  # 輸入梯度grads[batch,通道,高,寬]
        # 輸出特徵圖的每一點梯度的平均，並保持矩陣形狀為[batch,通道,1,1]
        return np.mean(grads, axis=(2, 3), keepdims=True)

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
            # output[i, target_category[i]]表示輸出層第i類的輸出值
            loss = loss + output[i, target_category[i]]
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
        # 取得輸入張量的寬度和高度
        width, height = input_tensor.size(-1), input_tensor.size(-2)
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

        # 循環遍歷每層的激活值和梯度值
        for layer_activations, layer_grads in zip(activations_list, grads_list):
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
        cam_per_target_layer = np.concatenate(
            cam_per_target_layer, axis=1)  # 將多層 CAM 進行水平連接
        cam_per_target_layer = np.maximum(
            cam_per_target_layer, 0)  # 取最大值，並消除負值
        result = np.mean(cam_per_target_layer, axis=1)  # 對每個目標進行平均
        return self.scale_cam_image(result)  # 對聚合後的 CAM 進行尺寸縮放

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []  # 創建一個空的列表 result 用來儲存處理後的影像
        for img in cam:  # 進入迴圈，對於 cam 中的每張影像執行以下操作
            img = img - np.min(img)  # 將影像中的最小值減去影像中的最小值，以進行正規化處理
            img = img / (1e-7 + np.max(img))  # 將影像除以影像中的最大值，同樣為了正規化處理
            if target_size is not None:  # 檢查目標尺寸是否為空值
                # 若有指定目標尺寸，使用 OpenCV (cv2) 進行影像縮放處理
                img = cv2.resize(img, target_size)
            result.append(img)  # 將處理後的影像加入 result 列表
        result = np.float32(result)  # 將 result 轉換成浮點數型態的 NumPy 陣列
        return result  # 回傳處理後的影像陣列

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:  # 判斷是否使用gpu
            input_tensor = input_tensor.cuda()  # 將PyTorch 張量 input_tensor 移動到 GPU 上

        # 正向传播得到网络输出logits(未经过softmax)
        # 將圖片輸入self.activations_and_grads()，執行class ActivationsAndGradients中的__call__，輸出神經網路輸出層的輸出值
        output = self.activations_and_grads(input_tensor)
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
            print(f"category id: {target_category}")  # 輸出最可能的類別
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()  # 清空历史梯度信息
        loss = self.get_loss(output, target_category)  # 獲取輸出層的輸出值，當作loss
        loss.backward(retain_graph=True)  # 把輸出值當loss作反向傳播

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(
            input_tensor)  # 獲取多層的gradcam
        return self.aggregate_multi_layers(cam_per_layer)  # 把多層gradcam做平均

    def __del__(self):  # 類別實例被銷毀時執行的方法。釋放相關資源
        self.activations_and_grads.release()

    def __enter__(self):  # 進入 with block 時執行的方法。返回實例本身
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):  # 離開 with block 時執行的方法，釋放相關資源
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):  # 檢查變數 exc_value 是否屬於 IndexError 類型
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

    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask), colormap)  # 將 CAM 遮罩應用成熱度圖
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


def main():
    # #如果使用自己的模型,需要导入自己的模型+预训练权重
    # #pretrained=True会自动的下载torchvision官方预训练好的权重
    # model = models.mobilenet_v3_large(pretrained=True)
    # #target_layers可以传入多个层结构，获取哪一个网络层结构的输出
    # target_layers = [model.features[-1]]
    # 使用vgg16网络
    img_size = 128  # 與訓練時相同的圖片尺寸
    num_classes = 2  # 貓和狗兩個類別
    # 定義CNNModel的物件，輸入通道數rgb為3、圖片大小、類別數
    model = CNNModel(input_shape=(3, img_size, img_size),
                     num_classes=num_classes)
    model_path = "model.pth"  # 設置權重文件的路徑，內包含訓練好的權重
    # torch.load()將文件轉成字典，再用load_state_dict()將字典中的權重載入model
    model.load_state_dict(torch.load(model_path))
    # 設置要計算gradcam的層，model.conv_layers[3]指nn.Conv2d(32, 64, kernel_size=3, padding=1)
    target_layers = [model.conv_layers]
    # print(target_layers)

   # model = models.vgg16(pretrained=True)
    #target_layers = [model.features]
    # print([model.features])

    # load image，读取的图片
    img_path = "4.jpg"  # 設置圖片路徑

    test_image = cv2.imread(img_path)  # cv2.imread()返回一個包含圖像rgb像素值的 NumPy 三維陣列
    # 調整圖片大小至(img_size, img_size)
    test_image = cv2.resize(test_image, (img_size, img_size))
    # OpenCV 在預設情況下讀取的圖像格式是 BGR，所以需要將其轉換為 RGB 格式
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_tensor = transforms.ToTensor()(test_image).unsqueeze(0)
    # transforms.ToTensor(): 這是一個 PyTorch 的轉換函式，它將圖像轉換成張量格式。它會將像素值範圍從 [0, 255] 轉換到 [0, 1]
    # .unsqueeze(0): 在張量的最前面（即第 0 維）添加一個維度。這是因為模型接受的輸入是一個批次（batch）的資料，所以需要在前面添加一個批次的維度。

    cam = GradCAM(model=model, target_layers=target_layers,
                  use_cuda=False)  # 定義GradCAM的物件，輸入 : 模型、目標層、是否使用gpu
    target_category = None  # 要計算哪個類別的gradcam，若是None，則計算機率最大的類別

    # 執行GradCAM中的__call__，獲取gradcam
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]  # 我們不需要一維資料，並取出gradcam
    # 丟進show_cam_on_image(測試圖片每個像素質都除以255,gradcam圖,圖片是否是rgb)
    visualization = show_cam_on_image(
        test_image / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)  # 顯示圖片
    plt.show()


main()
