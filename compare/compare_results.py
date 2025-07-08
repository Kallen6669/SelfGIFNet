import matplotlib.pyplot as plt
import numpy as np
import os

def read_training_stats(filename):
    """读取训练统计文件"""
    stats = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if '总训练时间:' in line:
                    stats['total_time'] = float(line.split(':')[1].split('秒')[0].strip())
                elif '平均每轮时间:' in line:
                    stats['avg_epoch_time'] = float(line.split(':')[1].split('秒')[0].strip())
                elif '本次实验总消耗显存:' in line:
                    # 新格式：本次实验总消耗显存: XXXMB
                    stats['gpu_consumed'] = float(line.split(':')[1].split('MB')[0].strip())
                elif '训练开始前GPU显存:' in line:
                    stats['initial_gpu_memory'] = float(line.split(':')[1].split('MB')[0].strip())
                elif '训练结束后GPU显存:' in line:
                    stats['final_gpu_memory'] = float(line.split(':')[1].split('MB')[0].strip())
                elif '最终GPU显存使用:' in line:
                    # 旧格式兼容
                    stats['gpu_memory'] = line.split(':')[1].strip()
                elif '最终系统内存使用:' in line:
                    stats['system_memory'] = line.split(':')[1].strip()
                elif 'Epoch' in line and '秒' in line:
                    if 'epoch_times' not in stats:
                        stats['epoch_times'] = []
                    time_str = line.split(':')[1].split('秒')[0].strip()
                    stats['epoch_times'].append(float(time_str))
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None
    return stats

def parse_memory_string(memory_str):
    """解析内存字符串，提取数值"""
    try:
        if 'MB' in memory_str:
            # 提取已使用内存
            used = float(memory_str.split('MB')[0].split('/')[0].strip())
            total = float(memory_str.split('MB')[0].split('/')[1].strip())
            return used, total, used/total*100
        elif 'GB' in memory_str:
            # 提取已使用内存
            used = float(memory_str.split('GB')[0].split('/')[0].strip())
            total = float(memory_str.split('GB')[0].split('/')[1].strip())
            return used*1024, total*1024, used/total*100
        else:
            return 0, 0, 0
    except:
        return 0, 0, 0

def create_comparison_report():
    """创建对比报告"""
    print("="*80)
    print("PyTorch vs Jittor ResNet18 训练性能对比")
    print("="*80)
    
    # 读取训练统计
    pytorch_stats = read_training_stats('pytorch_training_stats.txt')
    jittor_stats = read_training_stats('jittor_training_stats.txt')
    
    if pytorch_stats is None or jittor_stats is None:
        print("无法读取训练统计文件，请先运行训练脚本")
        return
    
    # 计算性能对比
    pytorch_time = pytorch_stats['total_time']
    jittor_time = jittor_stats['total_time']
    time_ratio = jittor_time / pytorch_time
    
    pytorch_avg_time = pytorch_stats['avg_epoch_time']
    jittor_avg_time = jittor_stats['avg_epoch_time']
    
    # 获取显存消耗数据
    pytorch_gpu_consumed = pytorch_stats.get('gpu_consumed', 0)
    jittor_gpu_consumed = jittor_stats.get('gpu_consumed', 0)
    
    # 打印对比结果
    print(f"\n📊 训练时间对比:")
    print(f"  PyTorch  总时间: {pytorch_time:.2f}秒 ({pytorch_time/3600:.2f}小时)")
    print(f"  Jittor   总时间: {jittor_time:.2f}秒 ({jittor_time/3600:.2f}小时)")
    print(f"  时间比率: {time_ratio:.2f}x (Jittor/PyTorch)")
    print(f"  性能差异: {((time_ratio-1)*100):.1f}%")
    
    print(f"\n📊 平均每轮时间对比:")
    print(f"  PyTorch  平均时间: {pytorch_avg_time:.2f}秒")
    print(f"  Jittor   平均时间: {jittor_avg_time:.2f}秒")
    print(f"  时间比率: {jittor_avg_time/pytorch_avg_time:.2f}x")
    
    print(f"\n📊 GPU显存消耗对比:")
    print(f"  PyTorch  显存消耗: {pytorch_gpu_consumed:.1f}MB")
    print(f"  Jittor   显存消耗: {jittor_gpu_consumed:.1f}MB")
    if pytorch_gpu_consumed > 0 and jittor_gpu_consumed > 0:
        memory_ratio = jittor_gpu_consumed / pytorch_gpu_consumed
        print(f"  显存比率: {memory_ratio:.2f}x")
        print(f"  显存差异: {((memory_ratio-1)*100):.1f}%")
    
    print(f"\n📊 系统内存使用对比:")
    print(f"  PyTorch  系统内存: {pytorch_stats['system_memory']}")
    print(f"  Jittor   系统内存: {jittor_stats['system_memory']}")
    
    # 性能评估
    print(f"\n🎯 性能评估:")
    if time_ratio < 1.1:
        print(f"  ✅ Jittor训练速度与PyTorch相当或更快")
    elif time_ratio < 1.5:
        print(f"  ⚠️  Jittor训练速度稍慢于PyTorch")
    else:
        print(f"  ❌ Jittor训练速度明显慢于PyTorch")
    
    if jittor_gpu_consumed > 0 and pytorch_gpu_consumed > 0:
        if jittor_gpu_consumed < pytorch_gpu_consumed * 1.1:
            print(f"  ✅ Jittor显存消耗与PyTorch相当或更少")
        elif jittor_gpu_consumed < pytorch_gpu_consumed * 1.5:
            print(f"  ⚠️  Jittor显存消耗稍多于PyTorch")
        else:
            print(f"  ❌ Jittor显存消耗明显多于PyTorch")
    
    # 创建可视化图表
    create_performance_charts(pytorch_stats, jittor_stats)
    
    print(f"\n📈 详细图表已保存到 performance_comparison.png")
    print("="*80)

def create_performance_charts(pytorch_stats, jittor_stats):
    """创建性能对比图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 总训练时间对比
    frameworks = ['PyTorch', 'Jittor']
    total_times = [pytorch_stats['total_time'], jittor_stats['total_time']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(frameworks, total_times, color=colors, alpha=0.7)
    ax1.set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_ylim(0, max(total_times) * 1.1)
    
    # 添加数值标签
    for bar, time in zip(bars1, total_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(total_times)*0.01,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. 平均每轮时间对比
    avg_times = [pytorch_stats['avg_epoch_time'], jittor_stats['avg_epoch_time']]
    bars2 = ax2.bar(frameworks, avg_times, color=colors, alpha=0.7)
    ax2.set_title('Average Epoch Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_ylim(0, max(avg_times) * 1.1)
    
    for bar, time in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_times)*0.01,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. 各轮训练时间趋势
    if 'epoch_times' in pytorch_stats and 'epoch_times' in jittor_stats:
        epochs = range(1, len(pytorch_stats['epoch_times']) + 1)
        ax3.plot(epochs, pytorch_stats['epoch_times'], 'o-', label='PyTorch', color='#FF6B6B', linewidth=2)
        ax3.plot(epochs, jittor_stats['epoch_times'], 's-', label='Jittor', color='#4ECDC4', linewidth=2)
        ax3.set_title('Training Time Trend by Epoch', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. GPU显存消耗对比
    pytorch_gpu_consumed = pytorch_stats.get('gpu_consumed', 0)
    jittor_gpu_consumed = jittor_stats.get('gpu_consumed', 0)
    
    gpu_consumed_values = [pytorch_gpu_consumed, jittor_gpu_consumed]
    bars4 = ax4.bar(frameworks, gpu_consumed_values, color=colors, alpha=0.7)
    ax4.set_title('GPU Memory Consumption Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_ylim(0, max(gpu_consumed_values) * 1.1 if max(gpu_consumed_values) > 0 else 100)
    
    # 添加数值标签
    for bar, memory in zip(bars4, gpu_consumed_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(gpu_consumed_values)*0.01 if max(gpu_consumed_values) > 0 else 5,
                f'{memory:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_comparison_report() 