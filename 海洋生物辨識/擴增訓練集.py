from PIL import Image, ImageEnhance
import os
import random
import shutil

# 定義原始訓練數據和目標數據文件夾
source_folder = 'traino'
target_folder = 'train_processed2'

# 如果目標文件夾不存在，則創建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍歷源文件夾中的子文件夾和圖像文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.jpg'):
            # 源文件路徑
            source_img_path = os.path.join(root, file)

            # 目標文件夾路徑
            target_subfolder = os.path.join(target_folder, os.path.relpath(root, source_folder))
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)

            # 目標文件路徑
            target_img_path = os.path.join(target_subfolder, file)

            # 複製原始圖像到目標文件夾
            shutil.copy2(source_img_path, target_img_path)

            # 讀取圖像
            img = Image.open(target_img_path)

            # 隨機旋轉
            random_angle = random.randint(0, 360)
            augmented_img1 = img.rotate(random_angle)
            augmented_img2 = img.rotate(random_angle)

            # 隨機縮放
            random_scale = random.uniform(0.8, 1.2)
            augmented_img1 = augmented_img1.resize((int(img.width * random_scale), int(img.height * random_scale)))
            augmented_img2 = augmented_img2.resize((int(img.width * random_scale), int(img.height * random_scale)))

            # 隨機調整亮度和對比度
            enhancer = ImageEnhance.Brightness(augmented_img1)
            augmented_img1 = enhancer.enhance(random.uniform(0.5, 1.5))
            enhancer = ImageEnhance.Contrast(augmented_img2)
            augmented_img2 = enhancer.enhance(random.uniform(0.5, 1.5))

            # 添加高斯噪聲
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = random.gauss(mean, sigma)
            augmented_img1 = ImageEnhance.Brightness(augmented_img1).enhance(1 + gauss)
            augmented_img2 = ImageEnhance.Brightness(augmented_img2).enhance(1 + gauss)

            # 將增強後的圖像保存到目標文件夾
            augmented_img1.save(target_img_path.replace('.jpg', '_aug1.jpg'))
            augmented_img2.save(target_img_path.replace('.jpg', '_aug2.jpg'))
