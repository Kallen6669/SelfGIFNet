import os
import sys
import jittor as jt
from jittor.optim import Adam
from jittor import transform
from jittor import models
from jittor import nn
from jittor.dataset import DataLoader

import argparse
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt

from args import Args as args
import jittor_msssim
from GIFNet_model import GIFNet
from GIFNetDataset import CustomDataset

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


def main():
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
    optimizer = Adam(GIFNet.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    # ssim_loss = jittor_msssim.msssim()
    ssim_loss = nn.MSELoss()


if __name__ == "__main__":
  main()