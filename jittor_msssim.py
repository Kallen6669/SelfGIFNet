import jittor as jt
from jittor import nn
from math import exp

def gaussian(window_size, sigma):
    gauss = jt.array([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = jt.unsqueeze(gaussian(window_size, 1.5), 1)
    # Jittor使用@操作符或jt.matmul进行矩阵乘法
    _2D_window = jt.unsqueeze(jt.unsqueeze((_1D_window @ _1D_window.transpose()).float32(), 0), 0)
    window = jt.expand(_2D_window, channel, 1, window_size, window_size)
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if jt.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if jt.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)

    mu1 = nn.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = nn.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = nn.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = nn.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = jt.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    weights = jt.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=jt.float32)
    levels = weights.shape[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = nn.avg_pool2d(img1, (2, 2))
        img2 = nn.avg_pool2d(img2, (2, 2))

    mssim = jt.stack(mssim)
    mcs = jt.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = jt.prod(pow1[:-1] * pow2[-1])
    return output


if __name__ == "__main__":
    img1 = jt.randn(1, 3, 256, 256)
    img2 = jt.randn(1, 3, 256, 256)
    print(ssim(img1, img2))
    print(msssim(img1, img2))