from PIL import Image
import os

def convert_images_to_rgb(input_folder, output_folder):
    # 如果輸出文件夾不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷輸入文件夾中的所有文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg'):
                # 原始圖像路徑
                input_img_path = os.path.join(root, file)

                # 目標圖像路徑
                output_img_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)

                # 讀取圖像
                img = Image.open(input_img_path)

                # 將圖像轉換為RGB模式
                img = img.convert("RGB")

                # 確保目標文件夾存在
                target_subfolder = os.path.dirname(output_img_path)
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)

                # 保存轉換後的圖像
                img.save(output_img_path)

# 使用範例
input_folder = 'blue_ringed_octopus_rgba'  # 輸入文件夾的路徑
output_folder = 'blue_ringed_octopus'  # 輸出文件夾的路徑

convert_images_to_rgb(input_folder, output_folder)
