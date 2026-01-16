import torch
import torch.nn.functional as F
import math

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
    
def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return 99.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

def ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    mu_x = sr.mean(dim=[2, 3])
    mu_y = hr.mean(dim=[2, 3])
    sigma_x = sr.var(dim=[2, 3], unbiased=False)
    sigma_y = hr.var(dim=[2, 3], unbiased=False)
    sigma_xy = ((sr - mu_x.unsqueeze(-1).unsqueeze(-1)) * (hr - mu_y.unsqueeze(-1).unsqueeze(-1))).mean(dim=[2, 3])
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean().item()

def lpips_metric(sr: torch.Tensor, hr: torch.Tensor, lpips_fn) -> float:
    sr = 2 * sr - 1
    hr = 2 * hr - 1
    with torch.no_grad():
        d = lpips_fn(sr, hr)
    return d.mean().item()