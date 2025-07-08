import jittor as jt
from jittor.optim import Adam
from jittor import models
from jittor import nn
from jittor.dataset.dataset import DataLoader

import argparse
import time
from tqdm import tqdm
import GPUtil
import os
import math

from args import Args as args
from jittor_msssim import msssim
from GIFNet_model import GIFNet
from GIFNetDataset import CustomDataset
from utils import lossChartSave, safe_weight_calculation, gradient, transform

# 启用CUDA
jt.flags.use_cuda = 1
# 使用CUDA 0卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置Jittor日志级别，抑制警告信息
os.environ['JITTOR_LOG_LEVEL'] = 'error'

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--trainDataRoot', type=str, default='./train_data', help='训练数据路径')
opt = parser.parse_args()


def main():
    # 记录训练开始时间
    start_time = time.time()
    
    print(f"Jittor CUDA状态: {jt.flags.use_cuda}")
    print(f"当前批次大小: {args.batch_size}")

    # 显示初始GPU内存使用情况
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            # 注意此处要与上述的os.environ['CUDA_VISIBLE_DEVICES'] = '0'一致
            gpu = gpus[0]
            print(f"初始GPU显存使用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    except:
        pass
    
    # 加载与训练好的densenet模型
    densenet = models.densenet121(pretrained=True)
    densenet.eval()
    features = list(densenet.features.children())
    features_1 = nn.Sequential(*features[:4])
    features_2 = nn.Sequential(*features[4:6])
    features_3 = nn.Sequential(*features[6:8])
    features_4 = nn.Sequential(*features[8:10])
    features_5 = nn.Sequential(*features[10:11])

    gifNet = GIFNet(args.s, args.n, args.channel, args.stride)
    optimizer = Adam(gifNet.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    ssim_loss = msssim

    root_dir = opt.trainDataRoot
    image_numbers = list(range(0, args.train_num))

    custom_dataset = CustomDataset(root = root_dir, image_numbers = image_numbers, transform = transform)
    # 优化数据加载器设置
    data_loader = DataLoader(custom_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=True,  # 启用随机打乱
                           num_workers=4,  # 增加工作进程数
                           drop_last=True)  # 丢弃不完整的批次
    # 开始训练
    print("开始训练")
    print(f"数据集大小: {len(data_loader)}")
    print(f"Batch大小: {args.batch_size}")
    print(f"总epoch数: {args.epochs}")
    print(f"学习率: {args.lr}")

    # 首先创建路径
    temp_path_model = os.path.join(args.save_fusion_model)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)
        
    temp_path_loss = os.path.join(args.save_loss_dir)
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)

    Loss_list_item1_spe = []
    Loss_list_item1_com = []
    Loss_list_item2_spe = []
    Loss_list_item2_com = []

    loss_item1_spe = 0.
    loss_item1_com = 0.
    loss_item2_spe = 0.
    loss_item2_com = 0.
    gifNet.train()

    step = 10

    # 初始化变量，避免未定义错误
    item1_IM_loss_cnn = None
    item2_clarity_loss = None
    e = 0  # 初始化epoch变量
    
    for e in range(args.epochs):
        batch_num = math.ceil(len(data_loader)/args.batch_size)
        loss_item1_spe = 0.
        loss_item2_spe = 0.
        loss_item1_com = 0.
        loss_item2_com = 0.        
        
        # 内层进度条：batch进度
        batch_pbar = tqdm(enumerate(data_loader), total=math.ceil(len(data_loader)/args.batch_size), 
                         desc=f"Epoch {e+1}/{args.epochs}", leave=True)
        
        for idx, batch in batch_pbar:
            try:
                # Jittor会自动管理设备，不需要手动指定设备
                # 数据已经是Jittor的Var类型，可以直接使用
                batch_ir, batch_vi, batch_ir_NF, batch_vi_FF = batch

                IVIF_step = 1;
                MFIF_step = 1;

                # 训练IVIF分支
                for _ in range(IVIF_step):
                    optimizer.zero_grad()
                    
                    # 预先计算特征，避免重复计算
                    fea_com_ivif = gifNet.forward_encoder(batch_ir, batch_vi)
                    with jt.no_grad():
                        fea_com_mfif = gifNet.forward_encoder(batch_ir_NF, batch_vi_FF)
                    
                    # 并行计算多个输出
                    out_rec = gifNet.forward_rec_decoder(fea_com_ivif)
                    fea_fused = gifNet.forward_MultiTask_branch(fea_com_ivif, fea_com_mfif, trainingTag = 1)
                    out_f = gifNet.forward_mixed_decoder(fea_com_ivif, fea_fused)

                    # 优化梯度计算，减少内存占用
                    with jt.no_grad():
                        # 使用更高效的方式计算梯度特征
                        t_batch_ir = batch_ir.clone()
                        t_batch_vi = batch_vi.clone()
                        dup_ir = jt.concat([t_batch_ir,t_batch_ir,t_batch_ir],1)
                        dup_vi = jt.concat([t_batch_vi,t_batch_vi,t_batch_vi],1)
                        layer1_feature_ir = features_1(dup_ir)
                        layer2_feature_ir = features_2(layer1_feature_ir)
                        layer3_feature_ir = features_3(layer2_feature_ir)
                        layer4_feature_ir = features_4(layer3_feature_ir)
                        layer5_feature_ir = features_5(layer4_feature_ir)
                        layer1_feature_vi = features_1(dup_vi)
                        layer2_feature_vi = features_2(layer1_feature_vi)
                        layer3_feature_vi = features_3(layer2_feature_vi)
                        layer4_feature_vi = features_4(layer3_feature_vi)
                        layer5_feature_vi = features_5(layer4_feature_vi)
                        # clamp防止极端值
                        layer1_feature_ir = jt.clamp(gradient(layer1_feature_ir)**2, 0, 100)
                        layer2_feature_ir = jt.clamp(gradient(layer2_feature_ir)**2, 0, 100)
                        layer3_feature_ir = jt.clamp(gradient(layer3_feature_ir)**2, 0, 100)
                        layer4_feature_ir = jt.clamp(gradient(layer4_feature_ir)**2, 0, 100)
                        layer5_feature_ir = jt.clamp(gradient(layer5_feature_ir)**2, 0, 100)
                        grad_ir_cnn = jt.mean(layer1_feature_ir)+jt.mean(layer2_feature_ir)+jt.mean(layer3_feature_ir)+jt.mean(layer4_feature_ir)+jt.mean(layer5_feature_ir)
                        grad_ir_cnn /= 5
                        layer1_feature_vi = jt.clamp(gradient(layer1_feature_vi)**2, 0, 100)
                        layer2_feature_vi = jt.clamp(gradient(layer2_feature_vi)**2, 0, 100)
                        layer3_feature_vi = jt.clamp(gradient(layer3_feature_vi)**2, 0, 100)
                        layer4_feature_vi = jt.clamp(gradient(layer4_feature_vi)**2, 0, 100)
                        layer5_feature_vi = jt.clamp(gradient(layer5_feature_vi)**2, 0, 100)
                        grad_vi_cnn = jt.mean(layer1_feature_vi) + jt.mean(layer2_feature_vi)+ jt.mean(layer3_feature_vi)+ jt.mean(layer4_feature_vi)+ jt.mean(layer5_feature_vi)
                        grad_vi_cnn /= 5
                        # 使用安全权重
                        weightNonInterestedIR_cnn, weightNonInterestedVI_cnn = safe_weight_calculation(grad_ir_cnn, grad_vi_cnn)

                    #item1
                    item1_IM_loss_cnn = weightNonInterestedIR_cnn*mse_loss(out_f, batch_ir) + weightNonInterestedVI_cnn*mse_loss(out_f,batch_vi)
                    item1_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi))
                    item1_IM_loss =  item1_IM_loss_cnn + item1_commonLoss
                    
                    # 检查损失值是否过大
                    if item1_IM_loss.item() > 1000:
                        print(f"警告: 第{idx}个batch的损失过大({item1_IM_loss.item():.2f})，跳过此batch")
                        continue
                    
                    # 检查损失是否为NaN
                    if jt.isnan(item1_IM_loss).any():
                        print(f"警告: 第{idx}个batch的损失为NaN，跳过此batch")
                        continue
                    
                    optimizer.step(item1_IM_loss)    
                loss_item1_spe += item1_IM_loss_cnn;
                loss_item1_com += item1_commonLoss;
        

                #MFIF branch
                for _idx in range(MFIF_step):
                    optimizer.zero_grad()                                

                    
                    fea_com = gifNet.forward_encoder(batch_ir_NF,batch_vi_FF)
                    with jt.no_grad():                
                        fea_com_ivif = gifNet.forward_encoder(batch_ir,batch_vi)
                        
                    out_rec = gifNet.forward_rec_decoder(fea_com)
                    
                    fea_fused = gifNet.forward_MultiTask_branch(fea_com_ivif = fea_com_ivif, fea_com_mfif = fea_com, trainingTag = 2);
                    
                    out_f = gifNet.forward_mixed_decoder(fea_com, fea_fused);                                                
                    
                    item2_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi));
                    item2_supLoss = mse_loss(out_f,batch_vi)
                    item2_clarity_loss = item2_supLoss + item2_commonLoss;
                    
                    # 检查损失值是否过大
                    if item2_clarity_loss.item() > 1000:
                        print(f"警告: 第{idx}个batch的MFIF损失过大({item2_clarity_loss.item():.2f})，跳过此batch")
                        continue
                    
                    # 检查损失值是否为NaN
                    if jt.isnan(item2_clarity_loss).any():
                        print(f"警告: 第{idx}个batch的MFIF损失为NaN，跳过此batch")
                        continue
                    
                    optimizer.step(item2_clarity_loss)                
                    

                loss_item2_spe += item2_supLoss;            
                loss_item2_com += item2_commonLoss;

            except Exception as exc:
                print(f"第{idx}个batch训练出错: {str(exc)}")
                # # 清理GPU内存
                # jt.gc()
                # # 强制清理内存
                # import gc
                # gc.collect()
                continue
                
            # 更新进度条描述
            if idx % step == 0:
                loss_item1_spe /= step
                loss_item2_spe /= step
                loss_item1_com /= step
                loss_item2_com /= step

                # 更新进度条描述
                batch_pbar.set_postfix({
                    'IM_loss': f'{item1_IM_loss_cnn.item():.4f}',
                    'MFIF_loss': f'{item2_clarity_loss.item():.4f}',
                    'LR': f'{args.lr:.2e}'
                })


                Loss_list_item1_spe.append(loss_item1_spe.item());
                Loss_list_item1_com.append(loss_item1_com.item());
                Loss_list_item2_spe.append(loss_item2_spe.item());
                Loss_list_item2_com.append(loss_item2_com.item());                

                loss_item1_spe = 0.
                loss_item2_spe = 0.
                loss_item1_com = 0.
                loss_item2_com = 0.

            if (idx+1) % 100 == 0:
                temp_loss = "epoch_" + str(e + 1) + "_batch_" + str(idx + 1) + \
                            "_block_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                lossChartSave(temp_loss,"item1_spe_loss",Loss_list_item1_spe);
                lossChartSave(temp_loss,"item1_com_loss",Loss_list_item1_com);
                lossChartSave(temp_loss,"item2_spe_loss",Loss_list_item2_spe);
                lossChartSave(temp_loss,"item2_com_loss",Loss_list_item2_com);
                

        # 打印epoch完成信息
        if item1_IM_loss_cnn is not None and item2_clarity_loss is not None:
            print(f"\nEpoch {e+1}/{args.epochs} 完成 - 最后损失: IM={item1_IM_loss_cnn.item():.4f}, MFIF={item2_clarity_loss.item():.4f}")
        else:
            print(f"\nEpoch {e+1}/{args.epochs} 完成 - 无有效损失数据")

        # 保存模型
        gifNet.eval()
        save_model_path = None
        if (e+1) % 5 == 0:
            save_model_filename = "MTFusion_net" + "_epoch_" + str(e + 1) + "_twoBranches"  + ".model"
            save_model_path = os.path.join(temp_path_model, save_model_filename)
            jt.save(gifNet.state_dict(), save_model_path)
            print(f"\nEpoch {e+1} 完成，模型已保存到: {save_model_path}")
        else:
            print(f"\nEpoch {e+1} 完成")
        ##############
        gifNet.train()

    # 计算训练总时间
    end_time = time.time()
    total_training_time = end_time - start_time
    
    # 获取最终显存使用情况
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # 获取第一个GPU
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            final_gpu_memory = f"{gpu_memory_used}MB / {gpu_memory_total}MB ({gpu_memory_percent:.1f}%)"
        else:
            final_gpu_memory = "无法获取GPU信息"
    except Exception as e:
        final_gpu_memory = f"获取GPU信息失败: {str(e)}"
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"训练总时间: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    print(f"平均每轮训练时间: {total_training_time/args.epochs:.2f} seconds")
    print(f"最终GPU显存使用情况: {final_gpu_memory}")
    print(f"模型保存路径: {save_model_path}")
    print("="*60)





if __name__ == "__main__":
  main()