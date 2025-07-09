#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估主控制脚本
按顺序运行Jittor评估、PyTorch评估和结果合并
"""

import os
import subprocess
import sys
import time

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"开始执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print("执行成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("模型性能评估脚本")
    print("="*50)
    
    # 检查必要文件是否存在
    required_files = [
        'evaluate_jittor.py',
        'evaluate_pytorch.py', 
        'merge_results.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误: 找不到必要文件 {file}")
            return
    
    # 步骤1: 运行Jittor模型评估
    print("\n步骤1: 评估Jittor模型")
    if not run_command("python evaluate_jittor.py", "Jittor模型评估"):
        print("Jittor模型评估失败，停止执行")
        return
    
    # 等待一下，确保GPU内存释放
    print("等待GPU内存释放...")
    time.sleep(5)
    
    # 步骤2: 运行PyTorch模型评估
    print("\n步骤2: 评估PyTorch模型")
    if not run_command("python evaluate_pytorch.py", "PyTorch模型评估"):
        print("PyTorch模型评估失败，停止执行")
        return
    
    # 步骤3: 合并结果
    print("\n步骤3: 合并评估结果")
    if not run_command("python merge_results.py", "结果合并"):
        print("结果合并失败")
        return
    
    print("\n" + "="*50)
    print("所有评估完成!")
    print("生成的文件:")
    print("- jittor_results.txt: Jittor模型评估结果")
    print("- pytorch_results.txt: PyTorch模型评估结果") 
    print("- model_comparison_results.txt: 合并后的完整对比结果")
    print("="*50)

if __name__ == '__main__':
    main() 