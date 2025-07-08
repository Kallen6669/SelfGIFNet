# PyTorch vs Jittor ResNet18 训练性能对比

这个项目用于对比PyTorch和Jittor框架在训练ResNet18模型时的性能差异，包括训练时间和显存占用。

## 📁 文件结构

```
compense/
├── checkpoints/               # 训练好的的模型
├── data/                      # 数据集
├── nohup/                     # 两次训练的log
├── args.py                    # 运行参数
├── jittor_training_stats      # jittor训练统计
├── pytorch_training_stats     # PyTorch训练统计
├── performance_comparison.png # 性能对比图表
├── train_pytorch.py           # PyTorch版本训练脚本
├── train_jittor.py            # Jittor版本训练脚本
├── compare_results.py         # 性能对比分析脚本
└── README.md                  # 说明文档
```

## 🚀 快速开始

### 运行训练
```bash
cd compare

# 运行PyTorch训练
python train_pytorch.py

# 运行Jittor训练
python train_jittor.py

# 生成对比报告
python compare_results.py
```

## 📊 训练配置

- **模型**: ResNet18s
- **数据集**: CIFAR10 
- **批次大小**: 32
- **训练轮数**: 50
- **优化器**: Adam
- **学习率**: 0.001
- **设备**: CUDA GPU

## 📈 监控指标

### 训练时间
- 总训练时间
- 平均每轮训练时间

### 显存使用
- GPU显存使用情况

## 📋 输出文件

训练完成后会生成以下文件：

- `pytorch_training_stats.txt` - PyTorch训练统计
- `jittor_training_stats.txt` - Jittor训练统计
- `performance_comparison.png` - 性能对比图表


### 可视化图表
1. **总训练时间对比柱状图**
2. **平均每轮时间对比柱状图**
3. **各轮训练时间趋势图**


## 🔧 环境要求

### 必需依赖
```bash
pip install torch torchvision
pip install jittor
pip install psutil GPUtil
pip install matplotlib numpy
pip install tqdm pillow
```

## 🛠️ 自定义配置

### 修改训练参数
在各自的训练脚本中修改以下参数：

```python
# 批次大小
batch_size = 32

# 训练轮数
epochs = 50

# 学习率
learning_rate = 0.001

# 优化器
optimizer = Adam(model.parameters(), lr=learning_rate)
```

### 修改模型配置
```python
# 修改模型结构
model = resnet18(num_classes=10)

# 修改损失函数
criterion = nn.CrossEntropyLoss()
```
