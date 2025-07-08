import scipy.io as scio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jittor.nn as nn
import jittor as jt
import os

from args import Args as args

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

EPSILON = 1e-5
def gradient(x):
	dim = x.shape;
	kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
	kernel = jt.unsqueeze(jt.unsqueeze(jt.array(kernel, dtype=jt.float32), 0), 0)
	kernel = jt.repeat(kernel, dim[1], dim[1], 1, 1);
	weight = kernel;
	pad = nn.ReflectionPad2d(1);		
	gradMap = nn.conv2d(pad(x), weight=weight, stride=1, padding=0);
	return gradMap;

def transform(img_array):
    # 格式转换并且归一化到0-1范围
    return jt.array(img_array) / 255.0