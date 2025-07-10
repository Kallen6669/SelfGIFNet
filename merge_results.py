"""
合并Jittor和PyTorch模型评估结果
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def merge_results():
    """
    合并两个模型的结果文件
    """
    jittor_file = 'jittor_results.txt'
    pytorch_file = 'pytorch_results.txt'
    output_file = 'model_comparison_results.txt'
    
    # 检查文件是否存在
    if not os.path.exists(jittor_file):
        print(f"错误: 找不到文件 {jittor_file}")
        return
    
    if not os.path.exists(pytorch_file):
        print(f"错误: 找不到文件 {pytorch_file}")
        return
    
    try:
        # 读取Jittor结果
        with open(jittor_file, 'r', encoding='utf-8') as f:
            jittor_content = f.read()
        
        # 读取PyTorch结果
        with open(pytorch_file, 'r', encoding='utf-8') as f:
            pytorch_content = f.read()
        
        # 创建合并结果
        merged_content = f"""
模型性能对比结果
================

{jittor_content}

{pytorch_content}

对比总结:
=========
"""
        # 保存合并结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        print(f"合并完成！结果已保存到 {output_file}")
    except Exception as e:
        print(f"合并过程中出现错误: {e}")
        
        # 生成可视化图表
    create_comparison_table(jittor_content, pytorch_content)

def extract_metrics(content):
    """从文本内容中提取指标数据"""
    metrics = {}
    
    # 提取运行时间
    time_match = re.search(r'运行时间:\s*([\d.]+)', content)
    if time_match:
        metrics['time'] = float(time_match.group(1))
    
    # 提取GPU内存使用
    memory_match = re.search(r'GPU内存使用:\s*([\d.]+)', content)
    if memory_match:
        metrics['memory'] = float(memory_match.group(1))
    
    # 提取EI指标
    ei_match = re.search(r'EI \(边缘强度\):\s*([\d.]+)', content)
    if ei_match:
        metrics['EI'] = float(ei_match.group(1))
    
    # 提取AG指标
    ag_match = re.search(r'AG \(平均梯度\):\s*([\d.]+)', content)
    if ag_match:
        metrics['AG'] = float(ag_match.group(1))
    
    # 提取VIF指标
    vif_match = re.search(r'VIF \(视觉信息保真度\):\s*([\d.]+)', content)
    if vif_match:
        metrics['VIF'] = float(vif_match.group(1))
    
    # 提取SCD指标
    scd_match = re.search(r'SCD \(结构相关性\):\s*([\d.]+)', content)
    if scd_match:
        metrics['SCD'] = float(scd_match.group(1))
    
    return metrics


def create_comparison_table(jittor_content, pytorch_content):
    """创建对比表格"""
    # 提取指标数据
    jittor_metrics = extract_metrics(jittor_content)
    pytorch_metrics = extract_metrics(pytorch_content)
    # 创建表格数据
    metrics_data = [
        ['Execution Time (s)', f"{jittor_metrics['time']:.4f}", f"{pytorch_metrics['time']:.4f}"],
        ['GPU Memory (MB)', f"{jittor_metrics['memory']:.2f}", f"{pytorch_metrics['memory']:.2f}"],
        ['EI (Edge Intensity)', f"{jittor_metrics['EI']:.4f}", f"{pytorch_metrics['EI']:.4f}"],
        ['AG (Average Gradient)', f"{jittor_metrics['AG']:.4f}", f"{pytorch_metrics['AG']:.4f}"],
        ['VIF (Visual Information Fidelity)', f"{jittor_metrics['VIF']:.4f}", f"{pytorch_metrics['VIF']:.4f}"],
        ['SCD (Structural Correlation Difference)', f"{jittor_metrics['SCD']:.4f}", f"{pytorch_metrics['SCD']:.4f}"]
    ]
    
    # 计算性能对比
    time_ratio = jittor_metrics['time'] / pytorch_metrics['time'] if pytorch_metrics['time'] > 0 else float('inf')
    memory_ratio = jittor_metrics['memory'] / pytorch_metrics['memory'] if pytorch_metrics['memory'] > 0 else float('inf')
    
    # 创建表格图片
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table_data = [['Metric', 'Jittor', 'PyTorch']] + metrics_data
    
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置表格标题
    plt.title('Model Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    # 保存表格
    plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
    print("对比表格已保存到 model_comparison_table.png")
    


def main():
    print("开始合并模型评估结果...")
    merge_results()

if __name__ == '__main__':
    main() 