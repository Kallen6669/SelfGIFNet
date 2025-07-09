import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.signal import correlate2d

def calc_EI(img):
    img = img.astype(np.float32)
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(edge_map)

def calc_AG(img):
    img = img.astype(np.float32)
    gy, gx = np.gradient(img)
    g = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(g)

def calc_VIF(fused, ir, vi):
    def vifp_single(ref, dist):
        ref = ref.astype(np.float32)
        dist = dist.astype(np.float32)
        sigma_nsq = 2.0
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(4):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            win = cv2.getGaussianKernel(N, sd)
            win = win @ win.T

            mu1 = convolve(ref, win, mode='reflect')
            mu2 = convolve(dist, win, mode='reflect')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = convolve(ref * ref, win, mode='reflect') - mu1_sq
            sigma2_sq = convolve(dist * dist, win, mode='reflect') - mu2_sq
            sigma12 = convolve(ref * dist, win, mode='reflect') - mu1_mu2

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[sv_sq <= 0] = eps

            num += np.sum(np.log10(1.0 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))

            ref = cv2.pyrDown(ref)
            dist = cv2.pyrDown(dist)

        return num / den if den != 0 else 1.0

    vif_ir = vifp_single(ir, fused)
    vif_vi = vifp_single(vi, fused)
    return max(vif_ir, vif_vi)

def calc_SCD(fused, ir, vi):
    """
    计算结构相关性差异 (Structural Correlation Difference)
    根据论文标准，SCD值应该在1左右
    """
    fused = fused.astype(np.float32)
    ir = ir.astype(np.float32)
    vi = vi.astype(np.float32)

    def correlation_coefficient(a, b):
        """计算两个图像之间的相关系数"""
        # 将图像展平为一维数组
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # 计算均值
        mean_a = np.mean(a_flat)
        mean_b = np.mean(b_flat)
        
        # 计算协方差和标准差
        numerator = np.sum((a_flat - mean_a) * (b_flat - mean_b))
        denominator = np.sqrt(np.sum((a_flat - mean_a)**2) * np.sum((b_flat - mean_b)**2))
        
        # 避免除零
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def structural_correlation_diff(a, b):
        """计算结构相关性差异"""
        # 计算相关系数
        corr_coef = correlation_coefficient(a, b)
        
        # 归一化到[0,1]范围，然后转换为差异度量
        # 相关系数范围是[-1,1]，我们将其映射到[0,1]
        normalized_corr = (corr_coef + 1) / 2
        
        # SCD = 1 - 归一化相关系数，这样值越小表示相关性越好
        scd = 1 - normalized_corr
        
        return scd

    # 计算融合图像与红外图像和可见光图像的结构相关性差异
    scd_ir = structural_correlation_diff(fused, ir)
    scd_vi = structural_correlation_diff(fused, vi)
    
    # 返回平均值
    return (scd_ir + scd_vi) / 2
