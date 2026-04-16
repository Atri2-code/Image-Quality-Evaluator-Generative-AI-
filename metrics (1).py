"""
metrics.py — No-reference and full-reference image quality metrics.

No-reference  : BRISQUE-inspired sharpness + noise estimation
Full-reference : SSIM, PSNR, FID (feature-level)
"""
import torch
import torch.nn.functional as F
import math
import numpy as np
from torchvision import transforms
from PIL import Image


def to_tensor(path_or_pil, size=256):
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    img = path_or_pil if isinstance(path_or_pil, Image.Image) else Image.open(path_or_pil).convert('RGB')
    return t(img).unsqueeze(0)


def psnr(real: torch.Tensor, generated: torch.Tensor) -> float:
    mse = F.mse_loss(real, generated).item()
    return round(10 * math.log10(1.0 / mse), 2) if mse > 0 else float('inf')


def ssim(a: torch.Tensor, b: torch.Tensor, k: int = 11) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(a, k, 1, k // 2)
    mu2 = F.avg_pool2d(b, k, 1, k // 2)
    m1s, m2s, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1  = F.avg_pool2d(a ** 2, k, 1, k // 2) - m1s
    s2  = F.avg_pool2d(b ** 2, k, 1, k // 2) - m2s
    s12 = F.avg_pool2d(a * b,  k, 1, k // 2) - m12
    num = (2 * m12 + C1) * (2 * s12 + C2)
    den = (m1s + m2s + C1) * (s1 + s2 + C2)
    return round(num.div(den).mean().item(), 4)


def sharpness(img: torch.Tensor) -> float:
    """Laplacian variance — proxy for perceived sharpness."""
    gray = img.mean(dim=1, keepdim=True)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return round(lap.var().item(), 4)


def fid_score(feats_real: torch.Tensor, feats_gen: torch.Tensor) -> float:
    """Simplified FID using mean/covariance of feature embeddings."""
    mu1, mu2 = feats_real.mean(0), feats_gen.mean(0)
    diff = mu1 - mu2
    cov1 = torch.cov(feats_real.T)
    cov2 = torch.cov(feats_gen.T)
    mean_diff = diff.dot(diff).item()
    cov_term  = torch.trace(cov1 + cov2 - 2 * torch.linalg.matrix_power(
        (cov1 @ cov2).clamp(min=0), 1
    )).item()
    return round(mean_diff + cov_term, 4)


def evaluate_pair(real_path: str, generated_path: str) -> dict:
    real = to_tensor(real_path)
    gen  = to_tensor(generated_path)
    return {
        'psnr':      psnr(real, gen),
        'ssim':      ssim(real, gen),
        'sharpness': sharpness(gen),
    }
