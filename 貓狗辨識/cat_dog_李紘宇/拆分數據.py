import os
import random
import shutil

# 设置随机种子
random.seed(42)

# 定义数据集目录
dataset_dir = 'dataset'

# 定义训练、验证、测试集的目录
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# 创建目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 定义拆分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 遍历数据集目录
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        random.shuffle(images)
        
        # 计算拆分数量
        train_split = int(train_ratio * len(images))
        val_split = int(val_ratio * len(images))
        
        # 拆分数据
        train_images = images[:train_split]
        val_images = images[train_split:train_split+val_split]
        test_images = images[train_split+val_split:]
        
        # 将图像复制到相应的目录
        for image in train_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(train_dir, category, image)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        
        for image in val_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(val_dir, category, image)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        
        for image in test_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(test_dir, category, image)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
