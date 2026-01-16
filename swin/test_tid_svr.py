import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import piq
from torch import nn
import warnings
warnings.filterwarnings("ignore")

from tid2019 import TID2019Dataset
from models.network_swinir import SwinIR
from performance import compute_metrics, format_metrics


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


def plot_scatter(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_metric(dataset, device, metric="psnr", model=None, pref_dir="tid_jpeg/PRefs"):
    """
    Extracts one metric alone: 'psnr', 'ssim', or 'lpips' between:
    1. PseudoRef ↔ Distorted (P↔D)
    2. Reference ↔ Distorted (R↔D)
    """
    values_pd = []  # pseudo-ref vs distorted
    values_rd = []  # reference vs distorted
    mos_list = []

    os.makedirs(pref_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for distorted_img, reference_img, mos, img_name in tqdm(loader, desc=f"Extracting {metric}"):
        distorted_img = distorted_img.to(device)
        reference_img = reference_img.to(device)
        pref_path = os.path.join(pref_dir, img_name[0])

        # Compute or load pseudo-reference
        if os.path.exists(pref_path):
            pseudo_ref = transforms.ToTensor()(plt.imread(pref_path)).unsqueeze(0).to(device)
        else:
            with torch.no_grad():
                pseudo_ref = model(distorted_img)
            pseudo_ref_img = torch.clamp(pseudo_ref, 0.0, 1.0)
            save_image(pseudo_ref_img, pref_path)

        # Compute metrics
        if metric == "psnr":
            val_pd = piq.psnr(distorted_img, pseudo_ref, data_range=1.).item()
            val_rd = piq.psnr(distorted_img, reference_img, data_range=1.).item()
        elif metric == "ssim":
            val_pd = piq.ssim(distorted_img, pseudo_ref, data_range=1.).item()
            val_rd = piq.ssim(distorted_img, reference_img, data_range=1.).item()
        elif metric == "lpips":
            val_pd = piq.LPIPS(reduction='none')(distorted_img, pseudo_ref).item()
            val_rd = piq.LPIPS(reduction='none')(distorted_img, reference_img).item()
        elif metric == "brisque":
            val_pd = piq.brisque(distorted_img, data_range=1., reduction='none').item()
            val_rd = piq.brisque(distorted_img, data_range=1., reduction='none').item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        values_pd.append(val_pd)
        values_rd.append(val_rd)
        mos_list.append(mos.item())

    return (np.array(values_pd).reshape(-1, 1), np.array(values_rd).reshape(-1, 1),
            np.array(mos_list))


def train_and_test_svr(X_train, y_train, X_test, y_test, metric_name, pair_type):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)

    try:
        metrics = compute_metrics(y_pred, y_test, fit_scale="logistic_5params")
        metrics = format_metrics(metrics)
        srcc = metrics.get("SROCC_2", 0)
        plcc = metrics.get("PLCC_2", 0)
        rmse = metrics.get("RMSE_2", 0)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        srcc = spearmanr(y_test, y_pred)[0]
        plcc = pearsonr(y_test, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- SVR Results ({pair_type}) for {metric_name.upper()} ---")
    print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, RMSE: {rmse:.4f}")

    os.makedirs("plots", exist_ok=True)
    plot_scatter(y_test, y_pred, xlabel="DMOS (True)", ylabel="Predicted DMOS",
                 title=f"SVR Predictions ({pair_type}, {metric_name.upper()})",
                 save_path=f"plots/svr_{pair_type}_{metric_name}.png")


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    # Create Training Set (80% of reference scenes)
    train_dataset = TID2019Dataset(
        root_dir='tid2019', 
        train=True,           # <--- Request Train Split
        train_ratio=0.8,      # 80% split
        seed=42,              # Fixed seed ensures no overlap with test
        transform=transform
    )

    # Create Test Set (The remaining 20% of reference scenes)
    test_dataset = TID2019Dataset(
        root_dir='tid2019', 
        train=False,          # <--- Request Test Split
        train_ratio=0.8,      # Must match train split ratio
        seed=42,              # Must match train seed
        transform=transform
    )

    # Build model
    model = build_model(device)
    ckpt_path = "SwinIR/experiments/swinir_jpeg/best_40.pth"
    state_model = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_model["state_dict"])

    for metric_name in ["brisque", "psnr", "ssim", "lpips"]:
        print(f"\nProcessing metric: {metric_name.upper()}")
        X_train_pd, X_train_rd, y_train = extract_metric(train_dataset, device, metric=metric_name, model=model, pref_dir="tid_jpeg/PRefs")
        X_test_pd, X_test_rd, y_test = extract_metric(test_dataset, device, metric=metric_name, model=model, pref_dir="tid_jpeg/PRefs")

        # SVR for PseudoRef ↔ Distorted
        train_and_test_svr(X_train_pd, y_train, X_test_pd, y_test, metric_name, pair_type="P-D")
        # SVR for Reference ↔ Distorted
        train_and_test_svr(X_train_rd, y_train, X_test_rd, y_test, metric_name, pair_type="R-D")


if __name__ == "__main__":
    main()
