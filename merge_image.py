import os
from PIL import Image
import numpy as np

# 文件夹路径
folder1 = 'submissions/vis_abbspo_12epoch'
folder2 = 'submissions/vis_h2rbox_4epoch'
output_folder = 'submissions/vis_contrast'

# 创建输出文件夹，如果不存在的话
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取两个文件夹中的 PNG 文件列表
files1 = set(f for f in os.listdir(folder1) if f.endswith('.png'))
files2 = set(f for f in os.listdir(folder2) if f.endswith('.png'))

# 找到两个文件夹中相同命名的 PNG 文件
common_files = files1.intersection(files2)

# 遍历所有相同命名的文件
for filename in common_files:
    # 打开两个文件夹中的图片
    img1 = Image.open(os.path.join(folder1, filename))
    img2 = Image.open(os.path.join(folder2, filename))

    # 确保图片的大小一致（如果不同，可以调整大小或裁剪）
    img1 = img1.resize(img2.size)  # 让两个图片大小一致

    # 创建一个新的图像，宽度是两张图像的宽度之和，高度不变
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_img = Image.new('RGB', (width1 + width2, max(height1, height2)))

    # 将图片粘贴到新图像中，添加一条亮线分隔
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width1, 0))

    # 创建一个亮线（例如，白色线条）
    line_thickness = 5
    new_img.paste((255, 255, 255), [width1, 0, width1 + line_thickness, new_img.height])

    # 保存合并后的图片
    new_img.save(os.path.join(output_folder, filename))

    # 打印当前处理的文件名
    print(f"Processed {filename}")

print("所有图片已处理并保存到", output_folder)
#左边是abbspo

# import os

# folder_path = '/mnt/nas/dataset_share/DOTA/split_ss_dota1_0/val/images'

# # 获取指定格式的图片数量
# image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

# print(f'There are {len(image_files)} .png images in the folder.')
