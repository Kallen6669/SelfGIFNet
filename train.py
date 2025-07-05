import os
import sys
import math

# 强制启用CUDA
os.environ['USE_CUDA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置Jittor日志级别，抑制警告信息
os.environ['JITTOR_LOG_LEVEL'] = 'error'

import jittor as jt
# 强制启用CUDA
jt.flags.use_cuda = 1

from jittor.optim import Adam
from jittor import transform
from jittor import models
import time
from jittor import nn
from jittor.dataset.dataset import DataLoader

import argparse
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from args import Args as args
from jittor_msssim import msssim
from GIFNet_model import GIFNet
from GIFNetDataset import CustomDataset
from utils import gradient

# 先确定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--trainDataRoot', type=str, default='./train_data', help='训练数据路径')
opt = parser.parse_args()

matplotlib.use('Agg')
# 显示损失图
def showLossChart(path,saveName):
    # 清空当前图像
    plt.cla()
    plt.clf()
    if (path == ""):
        return
    data = scio.loadmat(path)
    loss =data['Loss'][0]

    x_data = range(0,len(loss))
    y_data = loss

    plt.plot(x_data,y_data)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(saveName)

# 保存损失图
def lossChartSave(temp_loss,lossName,lossList):        
    # 保存item1_spe loss
    loss_filename_path = lossName + temp_loss
    save_loss_path = os.path.join(os.path.join(args.save_loss_dir), loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': lossList})
    showLossChart(save_loss_path,os.path.join(args.save_loss_dir)+"/"+lossName+'.png')        

# 安全的权重计算函数
def safe_weight_calculation(grad_ir, grad_vi):
    max_val = jt.maximum(grad_ir, grad_vi)
    exp_ir = jt.exp(grad_ir - max_val)
    exp_vi = jt.exp(grad_vi - max_val)
    total = exp_ir + exp_vi
    weight_ir = exp_ir / total
    weight_vi = exp_vi / total
    return weight_ir, weight_vi

def main():
    # 强制启用CUDA并设置内存管理
    jt.flags.use_cuda = 1
    # jt.flags.gpu_memory_limit = 0.7  # 限制GPU内存使用为70%
    
    print(f"Jittor CUDA状态: {jt.flags.use_cuda}")
    print(f"当前批次大小: {args.batch_size}")
    # print(f"GPU内存限制: {jt.flags.gpu_memory_limit}")
    
    # 加载与训练好的densenet模型
    densenet = models.densenet121(pretrained=True)
    densenet.eval()
    # Jittor会自动管理设备，不需要手动指定
    # 对于GPU，Jittor会自动检测并使用
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

    # 加载数据
    root_dir = opt.trainDataRoot
    image_numbers = list(range(1, args.train_num))

    # 与pytorch不同的地方，pytorch需要用到torchversion中的transforms
    def transform(img_array):
        # 归一化到0-1范围
        return jt.array(img_array) / 255.0

    custom_dataset = CustomDataset(root = root_dir, image_numbers = image_numbers, transform = transform)
    
    # 使用Jittor的方式设置batch_size和shuffle
    custom_dataset.set_attrs(batch_size=args.batch_size, shuffle=True)
    data_loader = custom_dataset
    # data_loader = DataLoader(custom_dataset)
    # 加载数据结束
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
    Loss_list_item1_learning = []
    Loss_list_item2_spe = []
    Loss_list_item2_com = []

    Loss_list_all = []

    viz_index = 0
    loss_item1_spe = 0.
    loss_item1_com = 0.
    loss_item1_learning = 0.
    loss_item2_spe = 0.
    loss_item2_com = 0.
    gifNet.train()

    step = 5

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
                    # 首先利用公共提取器来提取两个分支的特征,且MFIF分支的特征是冻结的
                    fea_com_ivif = gifNet.forward_encoder(batch_ir, batch_vi)
                    with jt.no_grad():
                        fea_com_mfif = gifNet.forward_encoder(batch_ir_NF, batch_vi_FF)
                    out_rec = gifNet.forward_rec_decoder(fea_com_ivif)
                    fea_fused = gifNet.forward_MultiTask_branch(fea_com_ivif, fea_com_mfif, trainingTag = 1)
                    out_f = gifNet.forward_mixed_decoder(fea_com_ivif, fea_fused); 

                     #计算源图像的信息量的度量。
                    with jt.no_grad():
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

                # mesg = "{}\t Count {} \t Epoch {}/{} \t Batch {}/{} \n " \
                #        "IM loss: {:.6f} \n". \
                #     format(time.ctime(), idx, e + 1, args.epochs, idx + 1, batch_num, item1_IM_loss_cnn.item())
                # print(mesg)

                Loss_list_item1_spe.append(loss_item1_spe.item());
                Loss_list_item1_com.append(loss_item1_com.item());
                Loss_list_item2_spe.append(loss_item2_spe.item());
                Loss_list_item2_com.append(loss_item2_com.item());                

                loss_item1_spe = 0.
                loss_item2_spe = 0.
                loss_item1_com = 0.
                loss_item2_com = 0.

            if (idx+1) % 10 == 0:
                temp_loss = "epoch_" + str(e + 1) + "_batch_" + str(idx + 1) + \
                            "_block_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                lossChartSave(temp_loss,"item1_spe_loss",Loss_list_item1_spe);
                lossChartSave(temp_loss,"item1_com_loss",Loss_list_item1_com);
                lossChartSave(temp_loss,"item2_spe_loss",Loss_list_item2_spe);
                lossChartSave(temp_loss,"item2_com_loss",Loss_list_item2_com);
                

            # if (idx+1) % 60 == 0:
            #     # save model ever 700 iter.
            #     #twoBranchesFusionModel.eval()
                
            #     save_model_filename = "MTFusion_net_epoch_" + str(e + 1) + "_count_" + str(idx+1) + "_twoBranches"  + ".model"
            #     save_model_path = os.path.join(temp_path_model, save_model_filename)
            #     jt.save(gifNet.state_dict(), save_model_path)
                
                
            #     print('Saving model at ' + save_model_path + '......')
                ##############
                #twoBranchesFusionModel.train()

        # 打印epoch完成信息
        if item1_IM_loss_cnn is not None and item2_clarity_loss is not None:
            print(f"\nEpoch {e+1}/{args.epochs} 完成 - 最后损失: IM={item1_IM_loss_cnn.item():.4f}, MFIF={item2_clarity_loss.item():.4f}")
        else:
            print(f"\nEpoch {e+1}/{args.epochs} 完成 - 无有效损失数据")

        # 保存模型
        gifNet.eval()
        save_model_path = None
        if e % 5 == 0:
            save_model_filename = "MTFusion_net" + "_epoch_" + str(e + 1) + "_twoBranches"  + ".model"
            save_model_path = os.path.join(temp_path_model, save_model_filename)
            jt.save(gifNet.state_dict(), save_model_path)
            print(f"\nEpoch {e+1} 完成，模型已保存到: {save_model_path}")
        else:
            print(f"\nEpoch {e+1} 完成")
        ##############
        gifNet.train()

    print("\n训练完成！")





if __name__ == "__main__":
  main()