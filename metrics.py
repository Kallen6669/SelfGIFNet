import numpy as np
import cv2
from scipy.signal import convolve2d
import math

# 边缘强度（Edge Intensity）
def calc_EI(img):
    img = img.astype(np.float32)
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.mean(edge_map)

# 平均梯度（Average Gradient）
def calc_AG(img):
    img = img.astype(np.float32)
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)
    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

# 视觉信息保真度（Visual Information Fidelity）
def calc_VIF(fused, ir, vi):
    return VIFF(fused, ir, vi)

# 结构相关性差异（Sum of Correlation Difference）
def calc_SCD(fused, ir, vi):
    def correlation_coefficient(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        mean_a = np.mean(a_flat)
        mean_b = np.mean(b_flat)
        numerator = np.sum((a_flat - mean_a) * (b_flat - mean_b))
        denominator = np.sqrt(np.sum((a_flat - mean_a) ** 2) * np.sum((b_flat - mean_b) ** 2))
        return 0.0 if denominator == 0 else numerator / denominator

    def structural_correlation_diff(a, b):
        corr_coef = correlation_coefficient(a, b)
        normalized_corr = (corr_coef + 1) / 2
        return 1 - normalized_corr

    scd_ir = structural_correlation_diff(fused, ir)
    scd_vi = structural_correlation_diff(fused, vi)
    return scd_ir + scd_vi

# ==================== 辅助：VIF 实现 ====================
def VIFF(image_F, image_A, image_B):
    refA = image_A.astype(np.float32)
    refB = image_B.astype(np.float32)
    dist = image_F.astype(np.float32)
    sigma_nsq = 2
    eps = 1e-10
    numA, denA, numB, denB = 0.0, 0.0, 0.0, 0.0

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        win = h / h.sum() if h.sum() != 0 else h

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')[::2, ::2]
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')[::2, ::2]
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

        def vif_terms(ref, mu1_sq, mu2, mu1_mu2):
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2 * mu2
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num = np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den = np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
            return num, den

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')

        mu1A_sq = mu1A * mu1A
        mu1B_sq = mu1B * mu1B
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2

        nA, dA = vif_terms(refA, mu1A_sq, mu2, mu1A_mu2)
        nB, dB = vif_terms(refB, mu1B_sq, mu2, mu1B_mu2)

        numA += nA
        denA += dA
        numB += nB
        denB += dB

    vifpA = numA / denA if denA != 0 else 1
    vifpB = numB / denB if denB != 0 else 1
    return vifpA + vifpB
