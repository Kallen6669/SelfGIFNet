import os
import numpy as np
from jittor.dataset.dataset import Dataset
from PIL import Image
from pathlib import Path
import jittor as jt
import random
from typing import List

class CustomDataset(Dataset):
    def __init__(self, root, image_numbers : List[int], transform=None):
        super().__init__()
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform
        # 设置Jittor数据集的基本属性
        self.total_len = len(self.image_numbers)

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "infrared/train/", f"{idx + 1}.jpg")
        vis_path = os.path.join(self.root, "visible/train/", f"{idx + 1}.jpg")
        visNF_path = os.path.join(self.root, "visible_focus_near/train/", f"{idx + 1}.jpg")
        visFF_path = os.path.join(self.root, "visible_focus_far/train/", f"{idx + 1}.jpg")
        # 使用灰度图是因为 便于统一格式
        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")
        visNF_img = Image.open(visNF_path).convert("L")
        visFF_img = Image.open(visFF_path).convert("L")

        # 转换为numpy数组
        ir_array = np.array(ir_img, dtype=np.float32)  
        vis_array = np.array(vis_img, dtype=np.float32) 
        visNF_array = np.array(visNF_img, dtype=np.float32) 
        visFF_array = np.array(visFF_img, dtype=np.float32) 

        # 添加通道维度 (H, W) -> (1, H, W)
        ir_array = ir_array[np.newaxis, :, :]
        vis_array = vis_array[np.newaxis, :, :]
        visNF_array = visNF_array[np.newaxis, :, :]
        visFF_array = visFF_array[np.newaxis, :, :]

        if self.transform:
            ir_array = self.transform(ir_array)
            vis_array = self.transform(vis_array)
            visNF_array = self.transform(visNF_array)
            visFF_array = self.transform(visFF_array)


        return ir_array, vis_array, visNF_array, visFF_array

# 测试dataset是否正确构建
if __name__ == "__main__":

    current_dir = os.getcwd()
    image_numbers = list(range(1, 18))
    # print(image_numbers)
    dataset = CustomDataset(root=os.path.join(current_dir, "train_data"), image_numbers=image_numbers)
        # 获取第一个样本并查看shape
    ir_img, vis_img, visNF_img, visFF_img = dataset[0]
    
    print(f"红外图像 shape: {ir_img.shape}")
    print(f"图像类型: {type(ir_img)}")    



    def my_transform(img_array):
        return jt.array(img_array)

    dataset = CustomDataset(root=os.path.join(current_dir, "train_data"), image_numbers=image_numbers, transform=my_transform)
    dataset.set_attrs(batch_size=8, shuffle=True)
    data_loader = dataset
    ir_img, vis_img, visNF_img, visFF_img = dataset[0]
    print(len(data_loader))
    print(f"红外图像 shape: {ir_img.shape}")
    # print(f"图像类型: {type(ir_img)}")    
    for batch in data_loader:
        ir_batch, vis_batch, visNF_batch, visFF_batch = batch
        print(f"  红外batch shape: {ir_batch.shape}")  # 这里才是batch的shape
    
    dataset.set_attrs(batch_size=1, shuffle=True)
    data_loader = dataset
    ir_img, vis_img, visNF_img, visFF_img = dataset[0]
    print(len(data_loader))
    print(f"红外图像 shape: {ir_img.shape}")
    # print(f"图像类型: {type(ir_img)}")    
    for batch in data_loader:
        ir_batch, vis_batch, visNF_batch, visFF_batch = batch
        print(f"  红外batch shape: {ir_batch.shape}")  # 这里才是batch的shape