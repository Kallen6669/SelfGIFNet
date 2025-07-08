import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from args import Args as args
import time
import psutil
import GPUtil
from tqdm import tqdm
import os
batch_size = args.batch_size
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR10 数据集
print("加载CIFAR10数据集...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# 模块构建
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer
        self.inplanes = 64


        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape([x.shape[0], -1])
        return self.fc(x)

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

# 初始化模型
model = resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    return running_loss / len(trainloader), 100.*correct/total

def get_gpu_memory():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[1]
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

# 创建保存模型的目录
os.makedirs('checkpoints', exist_ok=True)

# 主训练循环
print("="*60)
print("PyTorch ResNet18 训练开始")
print("="*60)
print(f"批次大小: 32")
print(f"训练轮数: 50")
print(f"数据集: CIFAR10")
print(f"优化器: Adam")
print(f"学习率: 0.001")
print("="*60)

start_time = time.time()
epoch_times = []

# 记录训练开始前的显存使用量
initial_gpu_memory = get_gpu_memory()
print(f"训练开始前GPU显存使用: {initial_gpu_memory}MB")
accuracy = 0
for epoch in range(args.epochs):
    print(len(trainloader))
    epoch_start_time = time.time()
    gpu_before = get_gpu_memory()
    ram_before = get_system_memory()
    loss, accuracy = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
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
        checkpoint_path = f'checkpoints/pytorch_resnet18_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, checkpoint_path)
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

with open('pytorch_training_stats.txt', 'w') as f:
    f.write(f"PyTorch ResNet18 训练统计\n")
    f.write(f"  准确率: {accuracy:.2f}%\n")
    f.write(f"总训练时间: {total_time:.2f}秒\n")
    f.write(f"平均每轮时间: {avg_time:.2f}秒\n")
    f.write(f"训练开始前GPU显存: {initial_gpu_memory}MB\n")
    f.write(f"训练结束后GPU显存: {final_gpu_memory}MB\n")
    f.write(f"本次实验总消耗显存: {total_gpu_consumed}MB\n")
    f.write(f"最终系统内存使用: {get_system_memory()}\n")
    f.write(f"各轮训练时间:\n")
    for i, t in enumerate(epoch_times):
        f.write(f"  Epoch {i+1}: {t:.2f}秒\n")
