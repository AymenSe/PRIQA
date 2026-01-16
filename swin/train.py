# ===============================
# Imports
# ===============================
from ast import List
import os
import random
from typing import Tuple, Optional

from jax import config
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data import DatasetJPEG
from models.network_swinir import SwinIR
from metrics import psnr, ssim, lpips_metric, AverageMeter
from config import Config

import warnings
warnings.filterwarnings("ignore")

# Optional LPIPS
try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    print("LPIPS not installed. `pip install lpips` to enable perceptual metric.")
    _HAS_LPIPS = False


# ===============================
# Utilities
# ===============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tensor_to_wandb_image(x: torch.Tensor):
    """
    x: (C,H,W) or (1,C,H,W), range [0,1]
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return wandb.Image(x)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1) as defined in the paper snippet.
    Formula: L = sqrt(||X - Y||^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        """
        Args:
            eps (float): Small constant for numerical stability. 
                         The paper suggests 10^-3.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): Predicted high-quality images (I_RHQ).
            targets (Tensor): Ground truth high-quality images (I_HQ).
        """
        diff = preds - targets
        # Calculate the loss element-wise
        # loss = sqrt( diff^2 + eps^2 )
        loss = torch.sqrt(diff * diff + self.eps ** 2)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
    
# ===============================
# Validation
# ===============================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: Optional[int] = None,
    lpips_fn=None,
    log_wandb: bool = True,
    max_log_images: int = 4,
):
    model.eval()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    lpips_m = AverageMeter() if lpips_fn is not None else None

    log_table = None
    logged = 0

    if log_wandb and wandb.run is not None:
        log_table = wandb.Table(columns=["LR", "HR", "SR"])

    for batch in tqdm(
        loader,
        desc=f"[Val] Epoch {epoch}" if epoch else "[Val]",
        leave=False,
        ncols=120
    ):
        lr = batch['L'].to(device)
        hr = batch['H'].to(device)

        sr = torch.clamp(model(lr), 0.0, 1.0)

        psnr_m.update(psnr(sr, hr), n=lr.size(0))
        ssim_m.update(ssim(sr, hr), n=lr.size(0))

        if lpips_fn is not None:
            lpips_m.update(lpips_metric(sr, hr, lpips_fn), n=lr.size(0))

        # ---- W&B IMAGE LOGGING (first few images only) ----
        if log_table is not None and logged < max_log_images:
            b = min(lr.size(0), max_log_images - logged)
            for i in range(b):
                log_table.add_data(
                    tensor_to_wandb_image(lr[i]),
                    tensor_to_wandb_image(hr[i]),
                    tensor_to_wandb_image(sr[i]),
                )
                logged += 1

        # Stop adding more images once we reached max_log_images
        if logged >= max_log_images:
            break

    # ---- Log the table to W&B ----
    if log_table is not None:
        wandb.log({f"val/images_epoch_{epoch}": log_table})

    return (
        psnr_m.avg,
        ssim_m.avg,
        lpips_m.avg if lpips_fn is not None else None
    )



# ===============================
# Builders
# ===============================
def build_model(device: torch.device) -> nn.Module:
    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=126,
        window_size=7,
        img_range=255.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv'
    )
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model.to(device)


def build_dataloaders(cfg):
    train_opt = {
        'dataroot_H': cfg.train_root,
        'phase': 'train',
        'n_channels': 3,
        'H_size': cfg.h_size,
        'quality_factor': cfg.qf_train,
        'quality_factor_test': cfg.qf_test,
        'is_color': True
    }

    test_opt = {
        'dataroot_H': cfg.test_root,
        'phase': 'test',
        'n_channels': 3,
        'H_size': cfg.h_size,
        'quality_factor': cfg.qf_train,
        'quality_factor_test': cfg.qf_test,
        'is_color': True
    }

    train_set = DatasetJPEG(train_opt)
    valid_set = DatasetJPEG(test_opt)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_set, valid_set, train_loader, valid_loader


# ===============================
# Training
# ===============================
def train(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    device = torch.device(f'cuda:{cfg.cuda_id}' if torch.cuda.is_available() else 'cpu')

    # ---- W&B ----
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config=cfg.__dict__,
        mode=cfg.wandb_mode
    )
    

    # ---- Data ----
    train_set, valid_set, train_loader, valid_loader = build_dataloaders(cfg)
    print(f"[INFO] Training samples: {len(train_set)}")
    print(f"[INFO] Validation samples: {len(valid_set)}")

    # ---- Model & Optim ----
    model = build_model(device)
    # criterion = nn.L1Loss()
    criterion = CharbonnierLoss(eps=1e-3)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )

    # ---- Tracking ----
    best_psnr = -1.0
    best_lpips = float('inf')
    ckpt_path = os.path.join(cfg.out_dir, f"best_all.pth") # TODO: {cfg.qf_train}

    # ---- Training Loop ----
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"[Train] Epoch {epoch}", ncols=120), 1
        ):
            lr = batch['L'].to(device)
            hr = batch['H'].to(device)

            optimizer.zero_grad(set_to_none=True)

            sr = torch.clamp(model(lr), 0.0, 1.0)
            loss = criterion(sr, hr)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            loss_meter.update(loss.item(), lr.size(0))

            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": step
            })

            if step % cfg.log_interval == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Step {step:05d} | "
                    f"Loss {loss_meter.avg:.4f}"
                )
                

        scheduler.step()

        # ---- Validation ----
        lpips_fn = None
        if _HAS_LPIPS:
            lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

        val_psnr, val_ssim, val_lpips = validate(
            model, valid_loader, device, epoch, lpips_fn
        )

        print(
            f"[Val] Epoch {epoch:03d} | PSNR {val_psnr:.3f} | SSIM {val_ssim:.4f}"
            + (f" | LPIPS {val_lpips:.4f}" if val_lpips is not None else "")
        )

        wandb.log({
            "val/psnr": val_psnr,
            "val/ssim": val_ssim,
            "val/lpips": val_lpips if val_lpips is not None else -1,
            "val/epoch": epoch
        })

        # ---- Checkpoint ----
        improved = (
            val_psnr > best_psnr or
            (val_lpips is not None and val_lpips < best_lpips)
        )

        if improved:
            best_psnr = max(best_psnr, val_psnr)
            if val_lpips is not None:
                best_lpips = min(best_lpips, val_lpips)

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': best_psnr,
                'lpips': best_lpips,
                'cfg': cfg.__dict__,
            }, ckpt_path)

            print(f"[INFO] Saved best checkpoint → {ckpt_path}")

    print(
        f"Training complete | Best PSNR: {best_psnr:.3f} dB | "
        f"Best LPIPS: {best_lpips:.4f}"
    )
    wandb.finish()


import argparse
import sys

# Assume train and other imports exist
# from test import ... 

def get_config():
    parser = argparse.ArgumentParser(description="JPEG Artifact Removal Configuration")

    # ---------------------------
    # Reproducibility
    # ---------------------------
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=2, help="GPU ID to use")

    # ---------------------------
    # Paths
    # ---------------------------
    parser.add_argument("--train_root", type=str, default="/home/asekhri@sic.univ-poitiers.fr/ICIP_2026/fivek_dataset/train")
    parser.add_argument("--test_root", type=str, default="/home/asekhri@sic.univ-poitiers.fr/ICIP_2026/fivek_dataset/test")
    parser.add_argument("--out_dir", type=str, default="./SwinIR/experiments/swinir_jpeg")

    # ---------------------------
    # Training
    # ---------------------------
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)

    # ---------------------------
    # JPEG settings
    # ---------------------------
    # parser.add_argument("--qf_train", type=List[int], default=[10], help="JPEG Quality Factor for training")
    # parser.add_argument("--qf_test", type=List[int], default=[10], help="JPEG Quality Factor for testing")
    
    parser.add_argument('--qf_train', nargs='+', type=int, default=[40])
    parser.add_argument('--qf_test', nargs='+', type=int, default=[40])

    parser.add_argument("--h_size", type=int, default=128, help="Patch size")

    # ---------------------------
    # Weights & Biases
    # ---------------------------
    parser.add_argument("--wandb_project", type=str, default="jpeg-artifact-removal")
    parser.add_argument("--wandb_name", type=str, default="swinir_dng_q10")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # This 'cfg' object now behaves exactly like your class, 
    # but contains values passed from the command line.
    cfg = get_config()
    
    print(f"Starting experiment: {cfg.wandb_name}")
    print(f"GPU: {cfg.cuda_id} | Batch: {cfg.batch_size} | LR: {cfg.lr}")
    
    # Pass the config to your train function
    train(cfg)