# 本文件用于对LLVIP数据集进行预处理，包括将图片像素转换，切割为小数据集等（我本地的数据集为3463张）以防爆显存

import os
import cv2
from PIL import Image
from pathlib import Path

num_images = 10
def detect_image_resolutions(folder_path):
    """
    检测文件夹内所有图片的分辨率，返回一个包含所有分辨率的set
    
    Args:
        folder_path (str): 图片文件夹路径
        
    Returns:
        set: 包含所有图片分辨率的集合，每个元素为 (width, height) 元组
    """
    resolutions = set()
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return resolutions
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为文件且扩展名支持
        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            if file_ext in supported_formats:
                try:
                    # 使用PIL读取图片获取分辨率
                    with Image.open(file_path) as img:
                        width, height = img.size
                        resolutions.add((width, height))
                        print(f"图片 {filename}: {width}x{height}")
                except Exception as e:
                    print(f"无法读取图片 {filename}: {e}")
    
    print(f"总共检测到 {len(resolutions)} 种不同的分辨率")
    return resolutions

def resize_images_by_half(input_folder, output_folder):
    """
    将输入文件夹中的图片在H和W两个维度上均缩小两倍，保存到输出文件夹
    
    Args:
        input_folder (str): 输入图片文件夹路径
        output_folder (str): 输出图片文件夹路径
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"输入文件夹 {input_folder} 不存在")
        return
    
    processed_count = 0
    error_count = 0
    cnt = 0
    for filename in os.listdir(input_folder):
        # 只生成300张图片
        if cnt == num_images:
            break
        cnt += 1
        
        file_path = os.path.join(input_folder, filename)
        
        # 检查是否为文件且扩展名支持
        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            if file_ext in supported_formats:
                try:
                    # 读取图片
                    img = cv2.imread(file_path)
                    if img is None:
                        print(f"无法读取图片 {filename}")
                        error_count += 1
                        continue
                    
                    # 获取原始尺寸
                    original_height, original_width = img.shape[:2]
                    
                    # 计算新尺寸（缩小两倍）
                    new_width = original_width // 2
                    new_height = original_height // 2
                    
                    # 调整图片大小
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 保存到输出文件夹
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, resized_img)
                    
                    print(f"处理完成: {filename} - 原始尺寸: {original_width}x{original_height} -> 新尺寸: {new_width}x{new_height}")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"处理图片 {filename} 时出错: {e}")
                    error_count += 1
    
    print(f"处理完成！成功处理 {processed_count} 张图片，失败 {error_count} 张")

if __name__ == "__main__":
    input_folder = "test_data/LLVIP_original/vi"  # 输入文件夹路径
    output_folder = "test_data/LLVIP/vi"  # 输出文件夹路径
    
    # # 检测分辨率
    # print("=== 检测图片分辨率 ===")
    # resolutions = detect_image_resolutions(input_folder)
    # print(f"检测到的分辨率: {resolutions}")
    
    # 缩小图片
    print("\n=== 缩小图片尺寸 ===")
    resize_images_by_half(input_folder, output_folder)
    input_folder = "test_data/LLVIP_original/ir"  # 输入文件夹路径
    output_folder = "test_data/LLVIP/ir"  # 输出文件夹路径
    # 缩小图片
    print("\n=== 缩小图片尺寸 ===")
    resize_images_by_half(input_folder, output_folder)
