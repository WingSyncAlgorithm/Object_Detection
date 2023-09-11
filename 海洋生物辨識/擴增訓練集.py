from PIL import Image, ImageEnhance
import os
import random

# 設置隨機種子
random.seed(42)

# 定義圖片處理函數
def process_image(input_path, output_dir, category, index):
    img = Image.open(input_path)
    
    # 隨機旋轉
    rotation_angle = random.randint(0, 360)
    rotated_img = img.rotate(rotation_angle)
    
    # 隨機縮放
    scale_factor = random.uniform(0.5, 2.0)
    scaled_img = rotated_img.resize((int(rotated_img.width * scale_factor), int(rotated_img.height * scale_factor)))  
    
    # 隨機翻轉
    if random.choice([True, False]):
        flipped_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_img = scaled_img
    
    # 隨機改變亮度和對比度
    brightness_factor = random.uniform(0.5, 2.0)
    contrast_factor = random.uniform(0.5, 2.0)
    
    enhancer = ImageEnhance.Brightness(flipped_img)
    bright_img = enhancer.enhance(brightness_factor)  
    
    enhancer = ImageEnhance.Contrast(bright_img)
    contrast_img = enhancer.enhance(contrast_factor)  
    
    output_path = os.path.join(output_dir, category)
    os.makedirs(output_path, exist_ok=True)

    contrast_img.save(os.path.join(output_path, str(index) + os.path.basename(input_path)))

# 設置train目錄和處理後的目錄
input_dir = 'train'
output_dir = 'train_processed'

# 創建處理後的目錄
os.makedirs(output_dir, exist_ok=True)

# 遍歷train目錄中的所有圖片並進行處理
for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)
    if os.path.isdir(category_path):
        for image in os.listdir(category_path):
            image_path = os.path.join(category_path, image)
            # 添加迴圈，輸出10張處理過的圖片
            for index in range(10):
                process_image(image_path, output_dir, category, index)
