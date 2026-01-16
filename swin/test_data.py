import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

# import your dataset
from data import DatasetJPEG


# ----------------------------
# Config
# ----------------------------
OPT = {
    'dataroot_H': '/home/asekhri@sic.univ-poitiers.fr/ICIP_2026/fivek_dataset/photos',
    'phase': 'test',                 # or 'train'
    'n_channels': 3,
    'H_size': 128,                   # not used in test
    'quality_factor': 10,
    'quality_factor_test': 10,
    'is_color': True
}

OUTPUT_ROOT = './dataset_outputs'
NUM_SAMPLES = 20        # how many samples to save
USE_DATALOADER = False  # True if you want batching


# ----------------------------
# Utilities
# ----------------------------
def tensor2uint(img):
    """
    Convert CHW float tensor [0,1] to HWC uint8.
    """
    img = img.clamp(0, 1)
    img = img.mul(255.0).byte()
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Main
# ----------------------------
def main():
    dataset = DatasetJPEG(OPT)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(os.path.join(OUTPUT_ROOT, 'L'))
    ensure_dir(os.path.join(OUTPUT_ROOT, 'H'))

    if USE_DATALOADER:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        iterator = enumerate(loader)
    else:
        iterator = enumerate(dataset)

    for i, data in iterator:
        if i >= NUM_SAMPLES:
            break

        if USE_DATALOADER:
            img_L = data['L'][0]
            img_H = data['H'][0]
            name = os.path.splitext(os.path.basename(data['H_path'][0]))[0]
        else:
            img_L = data['L']
            img_H = data['H']
            name = os.path.splitext(os.path.basename(data['H_path']))[0]

        img_L_np = tensor2uint(img_L)
        img_H_np = tensor2uint(img_H)

        # RGB → BGR for OpenCV
        if img_L_np.shape[2] == 3:
            img_L_np = cv2.cvtColor(img_L_np, cv2.COLOR_RGB2BGR)
            img_H_np = cv2.cvtColor(img_H_np, cv2.COLOR_RGB2BGR)

        out_L = os.path.join(OUTPUT_ROOT, 'L', f'{name}_{i:04d}_L.png')
        out_H = os.path.join(OUTPUT_ROOT, 'H', f'{name}_{i:04d}_H.png')

        cv2.imwrite(out_L, img_L_np)
        cv2.imwrite(out_H, img_H_np)

        print(f'Saved sample {i}:')
        print(f'  L -> {out_L}')
        print(f'  H -> {out_H}')

    print('Done.')


if __name__ == '__main__':
    main()
