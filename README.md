### 使用jittor框架完成GIFNet的复现
### 持续更新中！


#### 训练数据集
<img width="20" src="images/dataset1.png"> Training Set
- [Baidu Drive (code: x2i6)](https://pan.baidu.com/s/16lCjucwC476dFuxtfFbP3g?pwd=x2i6)

#### 训练代码
`python train.py --trainDataRoot ./train_data`

#### <img width="32" src="images/test.png"> Test

The **single required checkpoint** is avaiable in the folder "model"

<img width="20" src="images/set1.png"> Arguments:

```cpp
"--test_ir_root": Root path for the infrared input.
"--test_vis_root": Root path for the visible input.
"--VIS_IS_RGB": Visible input is stored in the RGB format.
"--IR_IS_RGB": Infrared input is stored in the RGB format.
"--save_path": Root path for the fused image.
```

<img width="20" src="images/task.png"> Infrared and Visible Image Fusion (IVIF):

```cpp
python test.py  --test_ir_root "images/IVIF/ir" --test_vis_root "images/IVIF/vis" --save_path "outputsIVIF" --VIS_IS_RGB 
```

<img width="20" src="images/task.png"> Multi-Focus Image Fusion (MFIF):

```cpp
python test.py  --test_ir_root "images/MFIF/nf" --test_vis_root "images/MFIF/ff" --save_path "outputsMFIF" --IR_IS_RGB --VIS_IS_RGB
```

<img width="20" src="images/task.png"> Multi-Exposure Image Fusion (MEIF):

```cpp
python test.py  --test_ir_root "images/MEIF/oe" --test_vis_root "images/MEIF/ue" --save_path "outputsMEIF" --IR_IS_RGB --VIS_IS_RGB 
```

<img width="20" src="images/task.png"> Medical Image Fusion:

```cpp
python test.py  --test_ir_root "images/Medical/pet" --test_vis_root "images/Medical/mri" --save_path "outputsMedical" --IR_IS_RGB
```

<img width="20" src="images/task.png"> Near-Infrared and Visible Image Fusion (NIR-VIS)

```cpp
python test.py  --test_ir_root "images/NIR-VIS/nir" --test_vis_root "images/NIR-VIS/vis" --save_path "outputsNIR-VIS" --VIS_IS_RGB
```

<img width="20" src="images/task.png"> Remote Sensing Image Fusion (Remote)

Step1 : Seprately fuse different bands of the multispectral image with the panchromatic image

(Python)
```cpp
python test.py  --test_ir_root "images/Remote/MS_band1" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand1"
python test.py  --test_ir_root "images/Remote/MS_band2" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand2"
python test.py  --test_ir_root "images/Remote/MS_band3" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand3"
python test.py  --test_ir_root "images/Remote/MS_band4" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand4"
```


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
