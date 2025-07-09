import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0
import cv2
import jittor as jt
import time
import numpy as np
from PIL import Image
import argparse
import GPUtil
from metrics import calc_EI, calc_AG, calc_VIF, calc_SCD
from GIFNet_model import GIFNet
from args import Args as args

parser = argparse.ArgumentParser()
parser.add_argument('--cpjittor', type=str, default='model/Final.model', help='fusion network weight')
parser.add_argument('--test_ir_root', type=str, default='test_data/LLVIP/ir', help='the test ir images root')
parser.add_argument('--test_vis_root', type=str, default='test_data/LLVIP/vi', help='the test vis images root')
parser.add_argument('--save_path_jittor', type=str, default='./outputs/outputsIVIF/jittor', help='the fusion results will be saved here')

opt = parser.parse_args()

def load_model(model_path):
    model = GIFNet(args.s, args.n, args.channel, args.stride)
    model.load_state_dict(jt.load(model_path))
    model.eval()
    return model

def run(model, ir_test_batch, vis_test_batch):
    img_ir = ir_test_batch
    img_vi = vis_test_batch

    fea_com = model.forward_encoder(img_ir, img_vi)    
    fea_fused = model.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com)            
    out_y_or_gray = model.forward_mixed_decoder(fea_com, fea_fused)    
    out_y_or_gray = out_y_or_gray[0,0,:,:].numpy()
    out_y_or_gray = out_y_or_gray*255
    return out_y_or_gray

def get_gpu_memory_usage():
    """获取GPU内存使用量（MB）"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # 获取第一个GPU
            return gpu.memoryUsed
        else:
            return 0
    except:
        return 0

def evaluate_metrics(fused_img, ir_img, vi_img):
    """评估融合图像的指标"""
    # 确保图像是灰度图
    if len(fused_img.shape) == 3:
        fused_gray = cv2.cvtColor(fused_img, cv2.COLOR_RGB2GRAY)
    else:
        fused_gray = fused_img
    
    if len(ir_img.shape) == 3:
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_RGB2GRAY)
    else:
        ir_gray = ir_img
        
    if len(vi_img.shape) == 3:
        vi_gray = cv2.cvtColor(vi_img, cv2.COLOR_RGB2GRAY)
    else:
        vi_gray = vi_img
    
    # 计算指标
    ei = calc_EI(fused_gray)
    ag = calc_AG(fused_gray)
    vif = calc_VIF(fused_gray, ir_gray, vi_gray)
    scd = calc_SCD(fused_gray, ir_gray, vi_gray)
    
    return ei, ag, vif, scd

def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image

def transform_jittor(img):
    img = jt.array(img).float32() / 255.0      
    img = img.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    img = jt.array(img).float32()
    return img

def main():
    jt.flags.use_cuda = 1
    model_path_jittor = opt.cpjittor
    output_path_jittor = opt.save_path_jittor

    if os.path.exists(output_path_jittor) is False:
        os.mkdir(output_path_jittor)
    
    ir_path_root = opt.test_ir_root
    vis_path_root = opt.test_vis_root
    names = os.listdir(ir_path_root)
    
    # 初始化指标统计
    jittor_metrics = {'EI': [], 'AG': [], 'VIF': [], 'SCD': []}
       
    time_jittor = time.time()
    gpu_memory_used_start = get_gpu_memory_usage()
    
    # 测试jittor的性能
    with jt.no_grad():
        model_jittor = load_model(model_path_jittor)

        for fileName in names:        
            ir_path = os.path.join(ir_path_root, fileName)
            vis_path = os.path.join(vis_path_root, fileName)            
            # 灰度图
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) # (H,W)array
            # RGB 转 YCbCr
            vis_img = Image.open(vis_path).convert("RGB")
            vis_img, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vis_img) # 三个全是(H,W)的array
            vis_img = vis_img.astype(np.uint8)

            ir_img_jittor = transform_jittor(ir_img)
            vis_img_jittor = transform_jittor(vis_img)
            
            fused_y_jittor = run(model_jittor, ir_img_jittor, vis_img_jittor)
            outputFuse_path_jittor = os.path.join(output_path_jittor, fileName)
            
            #如果最终结果是彩色图像
            fuseImage_jittor = ycbcr_to_rgb(fused_y_jittor, vi_img_cb, vi_img_cr)
            fuseImage_jittor.save(outputFuse_path_jittor)
            
            # 评估指标
            fused_array = np.array(fuseImage_jittor)
            ir_array = cv2.imread(ir_path)
            vis_array = np.array(Image.open(vis_path).convert("RGB"))
            
            ei, ag, vif, scd = evaluate_metrics(fused_array, ir_array, vis_array)
            jittor_metrics['EI'].append(ei)
            jittor_metrics['AG'].append(ag)
            jittor_metrics['VIF'].append(vif)
            jittor_metrics['SCD'].append(scd)
                
    time_jittor = time.time() - time_jittor
    gpu_memory_used_end = get_gpu_memory_usage()
    jittor_memory_used = gpu_memory_used_end - gpu_memory_used_start

    # 计算平均指标
    jittor_avg_metrics = {key: np.mean(values) for key, values in jittor_metrics.items()}
    
    # 保存结果到txt文件
    results = f"""
Jittor模型性能评估结果
=====================

运行时间: {time_jittor:.4f} 秒
GPU内存使用: {jittor_memory_used:.2f} MB
EI (边缘强度): {jittor_avg_metrics['EI']:.4f}
AG (平均梯度): {jittor_avg_metrics['AG']:.4f}
VIF (视觉信息保真度): {jittor_avg_metrics['VIF']:.4f}
SCD (结构相关性): {jittor_avg_metrics['SCD']:.4f}

测试图片数量: {len(names)} 张
"""
    
    # 保存到文件
    with open('jittor_results.txt', 'w', encoding='utf-8') as f:
        f.write(results)
    
    print(results)
    print(f"Jittor模型结果已保存到 jittor_results.txt")
            
if __name__ == '__main__':
    main() 