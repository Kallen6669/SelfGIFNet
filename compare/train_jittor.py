import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam
import time
import psutil
import GPUtil
from args import Args as args
import math
import pickle
from jittor.dataset.dataset import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from jittor.dataset.cifar import CIFAR10
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jt.flags.use_cuda = 1
# jt.flags.log_silent = 1
# jt.flags.profiler_enable = 0

# 数据加载器
def transform(image):
    # 转换为numpy数组并归一化
    img_array = np.array(image).astype(np.float32) / 255.0
    # 标准化
    mean = np.array([0.5, 0.5, 0.5],dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5],dtype=np.float32)
    img_array = (img_array - mean) / std
    return jt.array(img_array)

print("加载CIFAR10数据集...")

batch_size = args.batch_size
trainset = CIFAR10(root='./data',train=True, transform=transform,download=False)
trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=2)
# ResNet18 组件
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride, bias=False, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, 
                  norm_layer=nn.BatchNorm2d):
        super().__init__()

        self._norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape([x.shape[0], -1])
        return self.fc(x)

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 模型、损失、优化器
model = resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def get_gpu_memory():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return gpu.memoryUsed  # 只返回使用的显存MB数
        else:
            return "无法获取GPU信息"
    except Exception as e:
        print(f"获取GPU信息失败: {str(e)}")
        return 0

def get_system_memory():
    try:
        memory = psutil.virtual_memory()
        return f"{memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB ({memory.percent:.1f}%)"
    except Exception as e:
        return f"获取系统内存信息失败: {str(e)}"

# 训练函数
def train_epoch(model, trainloader, criterion, optimizer, epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}",total=math.ceil(len(trainloader)/batch_size))
    for batch_idx, (data, target) in enumerate(pbar):
        
        optimizer.zero_grad()
        data = data.permute(0, 3, 1, 2)
        output = model(data)
        loss = criterion(output, target)
        optimizer.step(loss)
        running_loss += float(loss.numpy())
        pred = jt.argmax(output, dim=1)
        total += target.shape[0]
        correct += int((pred[0].equal(target)).sum().numpy())
        pbar.set_postfix({
            "Loss": f"{float(loss.numpy()):.4f}",
            "Acc": f"{100. * correct / total:.2f}%"
        })
    return running_loss / len(trainloader), 100. * correct / total

# 创建保存模型的目录
os.makedirs('checkpoints', exist_ok=True)

# 主训练循环
print("="*60)
print("Jittor ResNet18 训练开始")
print("="*60)
print(f"批次大小: 8")
print(f"训练轮数: 50")
print(f"数据集: CIFAR10")
print(f"优化器: Adam")
print(f"学习率: 0.001")
print("="*60)

# 主循环
start_time = time.time()
epoch_times = []

# 记录训练开始前的显存使用量
initial_gpu_memory = get_gpu_memory()
print(f"训练开始前GPU显存使用: {initial_gpu_memory}MB")
accuracy = 0
# trainloader = jt.dataset.DataLoader(trainset, batch_size=32, num_workers=2, shuffle=True)
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    gpu_before = get_gpu_memory()
    ram_before = get_system_memory()
    loss, accuracy = train_epoch(model, trainloader, criterion, optimizer, epoch)
    gpu_after = get_gpu_memory()
    ram_after = get_system_memory()
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    # 计算本次实验消耗的显存（相对于训练开始前）
    gpu_consumed = gpu_after - initial_gpu_memory

    print(f"\nEpoch {epoch+1}/50 完成:")
    print(f"  损失: {loss:.4f}")
    print(f"  准确率: {accuracy:.2f}%")
    print(f"  Epoch时间: {epoch_time:.2f}秒")
    print(f"  当前GPU显存: {gpu_after}MB")
    print(f"  本次实验消耗显存: {gpu_consumed}MB")
    print(f"  RAM: {ram_before} → {ram_after}")
    print("-" * 60)

    # 每五轮保存一次模型
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f'checkpoints/jittor_resnet18_epoch_{epoch+1}.pkl'
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"模型已保存到: {checkpoint_path}")

total_time = time.time() - start_time
avg_time = sum(epoch_times) / len(epoch_times)
final_gpu_memory = get_gpu_memory()
total_gpu_consumed = final_gpu_memory - initial_gpu_memory

print("\n" + "="*60)
print("训练完成！")
print("="*60)
print(f"总时间: {total_time:.2f}s ({total_time/3600:.2f}小时)")
print(f"平均每轮: {avg_time:.2f}s")
print(f"训练开始前GPU显存: {initial_gpu_memory}MB")
print(f"训练结束后GPU显存: {final_gpu_memory}MB")
print(f"本次实验总消耗显存: {total_gpu_consumed}MB")
print(f"最终系统内存: {get_system_memory()}")
print("="*60)

with open('jittor_training_stats.txt', 'w') as f:
    f.write(f"Jittor ResNet18 训练统计\n")
    print(f"  准确率: {accuracy:.2f}%")
    f.write(f"总训练时间: {total_time:.2f}秒\n")
    f.write(f"平均每轮时间: {avg_time:.2f}秒\n")
    f.write(f"训练开始前GPU显存: {initial_gpu_memory}MB\n")
    f.write(f"训练结束后GPU显存: {final_gpu_memory}MB\n")
    f.write(f"本次实验总消耗显存: {total_gpu_consumed}MB\n")
    f.write(f"最终系统内存使用: {get_system_memory()}\n")
    f.write(f"各轮训练时间:\n")
    for i, t in enumerate(epoch_times):
        f.write(f"  Epoch {i+1}: {t:.2f}秒\n")