import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU 1
import cv2
import jittor as jt
import time
import numpy as np
from GIFNet_model import GIFNet
from args import Args as args
import utils
import matplotlib.pyplot as plt  
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='model/Final.model', help='fusion network weight')

parser.add_argument('--test_ir_root', type=str, required=True, help='the test ir images root')
parser.add_argument('--IR_IS_RGB', action='store_true', help='The IR input is stored in RGB format')

parser.add_argument('--test_vis_root', type=str, required=True, help='the test vis images root')
parser.add_argument('--VIS_IS_RGB', action='store_true', help='The VIS input is stored in RGB format')

parser.add_argument('--save_path', type=str, default='./outputs/', help='the fusion results will be saved here')

opt = parser.parse_args()

def resize_images(images, target_size=(128, 128)):
    return jt.nn.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

def load_model(model_path_twoBranches):
    # 设置内存管理
    jt.flags.use_cuda = 1
    
    model = GIFNet(args.s, args.n, args.channel, args.stride)

    model.load_state_dict(jt.load(model_path_twoBranches))

    para = sum([np.prod(list(p.shape)) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model.__class__.__name__, para * type_size / 1000 / 1000))
    
    total = sum([param.numel() for param in model.parameters()])
    print('Number of parameter: {:4f}M'.format(total / 1e6))
    
    model.eval()

    return model

def run(model, ir_test_batch, vis_test_batch, output_path, img_name):

    img_ir = ir_test_batch
    img_vi = vis_test_batch

    # 添加维度检查
    print(f"Input IR shape: {img_ir.shape}")
    print(f"Input VIS shape: {img_vi.shape}")
    
    # 确保输入是4维张量
    if img_ir.ndim != 4:
        raise ValueError(f"Expected 4D tensor for IR, got {img_ir.ndim}D")
    if img_vi.ndim != 4:
        raise ValueError(f"Expected 4D tensor for VIS, got {img_vi.ndim}D")

    #print(img_ir.dtype)
    #print(img_vi.dtype)
    fea_com = model.forward_encoder(img_ir, img_vi)    
    fea_fused = model.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com)            
    out_y_or_gray = model.forward_mixed_decoder(fea_com, fea_fused);    
    
    out_y_or_gray = out_y_or_gray[0,0,:,:].numpy()
    out_y_or_gray = out_y_or_gray*255
    
    return out_y_or_gray


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

def fuse_cb_cr(Cb1,Cr1,Cb2,Cr2):
    H, W = Cb1.shape
    Cb = np.ones((H, W),dtype=np.float32)
    Cr = np.ones((H, W),dtype=np.float32)

    for k in range(H):
        for n in range(W):
            if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                Cb[k, n] = 128
            else:
                middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                Cb[k, n] = middle_1 / middle_2

            if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                Cr[k, n] = 128
            else:
                middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                Cr[k, n] = middle_3 / middle_4
    return Cb, Cr

def main():

    # 原文中这几行没用到
    # test_path = "./images/"
    # imgs_paths_ir, names = utils.list_images(test_path)
    # num = len(imgs_paths_ir)
    jt.flags.use_cuda = 1
    model_path_twoBranches = opt.checkpoint

    output_path = opt.save_path

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
       
    with jt.no_grad():
        model = load_model(model_path_twoBranches)
        
        ir_path_root = opt.test_ir_root
        vis_path_root = opt.test_vis_root
        
        names = os.listdir(ir_path_root)
        
        for fileName in names:        
            ir_path = os.path.join(ir_path_root, fileName)
            vis_path = os.path.join(vis_path_root, fileName)            

            #红外输入是RGB
            if opt.IR_IS_RGB:
                ir_img = Image.open(ir_path).convert("RGB")
                ir_img, ir_img_cb, ir_img_cr = rgb_to_ycbcr(ir_img)               
                ir_img = ir_img.astype(np.uint8)
            else:    
                ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) # (1024,1280)的array
            
            #可见光输入是RGB            
            if opt.VIS_IS_RGB:    
                vis_img = Image.open(vis_path).convert("RGB")
                vis_img, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vis_img) # 三个全是(1024,1280)的array
                vis_img = vis_img.astype(np.uint8)
            else:
                vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            
            #只要有一个是RGB就要做cb和cr的融合
            if opt.IR_IS_RGB or opt.VIS_IS_RGB:
                #都是RGB，两者融合
                if opt.IR_IS_RGB and opt.VIS_IS_RGB:
                    vi_img_cb, vi_img_cr = fuse_cb_cr(vi_img_cb, vi_img_cr, ir_img_cb, ir_img_cr);
                elif  opt.IR_IS_RGB:
                #ir是rgb，换成ir的
                    vi_img_cb = ir_img_cb
                    vi_img_cr = ir_img_cr
                #否则，默认保留可见光的cb和cr.                
            
            # 将numpy数组转换为jittor tensor
            ir_img = jt.array(ir_img).float32() / 255.0
            vis_img = jt.array(vis_img).float32() / 255.0        
            
            # 确保张量是4维的 (batch_size, channels, height, width)
            if ir_img.ndim == 2:
                ir_img = ir_img.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            elif ir_img.ndim == 3:
                ir_img = ir_img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

            if vis_img.ndim == 2:
                vis_img = vis_img.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            elif vis_img.ndim == 3:
                vis_img = vis_img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

            ir_test_batch = ir_img.float32()
            vis_test_batch = vis_img.float32()
            
            # 添加维度检查
            print(f"IR tensor shape: {ir_test_batch.shape}")
            print(f"VIS tensor shape: {vis_test_batch.shape}")
            
            fused_y_or_gray = run(model, ir_test_batch, vis_test_batch, output_path, fileName)
            
            outputFuse_path = os.path.join(output_path, fileName)
            
            #如果最终结果是彩色图像
            if opt.IR_IS_RGB or opt.VIS_IS_RGB:
                fuseImage = ycbcr_to_rgb(fused_y_or_gray, vi_img_cb, vi_img_cr);        
                fuseImage.save(outputFuse_path);        
            else:
                fused_y_or_gray = fused_y_or_gray.astype(np.uint8)
                #print(fused_y_or_gray)
                cv2.imwrite(outputFuse_path, fused_y_or_gray)
                
            print('Image -> '+ fileName + ' Done......')    
            
if __name__ == '__main__':
    main()
