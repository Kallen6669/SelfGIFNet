### 使用jittor框架完成GIFNet的复现
### 持续更新中！


#### 训练数据集
<img width="20" src="images/dataset1.png"> Training Set
- [Baidu Drive (code: x2i6)](https://pan.baidu.com/s/16lCjucwC476dFuxtfFbP3g?pwd=x2i6)

#### 训练代码
`python train.py --trainDataRoot ./train_data`

#### 遇到的问题
1. 
 - 报错：显示  `version `GLIBCXX_3.4.30' not found  
 - 原因： 表示 Jittor 编译生成的 .so 文件依赖了 libstdc++.so.6 中的 GLIBCXX_3.4.30 版本符号，但你的当前环境中没有该版本的 libstdc++
 - 解决方案 ：在 Conda 环境中升级 gcc/libstdc++，不会影响系统环境`conda install -c conda-forge gcc`
2. 
 - 报错：raise RuntimeError(f"MD5 mismatch between the server and the downloaded file {file_path}")
 - 原因：Jittor 在下载 CUTLASS 库时，文件校验失败（MD5 不匹配）
 - 解决方案： 从网上把.zip文件复制下来直接放到相应位置了 可以参考`https://discuss.jittor.org/t/topic/936`
 - 文件 - [cutlass](https://pan.baidu.com/s/1OycxDBUy8d_bv5L0avnuTg?pwd=73oj)
 3.
 - 报错：`CUDA found but cudnn is not loaded`
 - 原因：找到了cuda但是没有正确加载
 - 解决方案：使用官方推荐的命令`python -m jittor_utils.install_cuda`,之后改变环境变量`export LD_LIBRARY_PATH=/root/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/lib64:$LD_LIBRARY_PATH`（下载完成后需要等待一段时间，请耐心，强制关闭可能会出问题）