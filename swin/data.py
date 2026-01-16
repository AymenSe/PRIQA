import os
import glob
import random
import numpy as np
import cv2
import rawpy
import torch
import torch.utils.data as data


# ===============================
# Utilities
# ===============================

def read_dng(path):
    """Read a DNG (RAW) image and return RGB uint8."""
    with rawpy.imread(path) as raw:
        img = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8
        )
    return img  # RGB uint8


def uint2tensor3(img):
    """Convert HWC uint8 image to CHW float tensor in [0,1]."""
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0


def augment_img(img, mode):
    """Flip / rotate augmentation (8 modes)."""
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.rot90(img, 2)
    elif mode == 4:
        return np.rot90(img, 3)
    elif mode == 5:
        return np.flipud(np.rot90(img))
    elif mode == 6:
        return np.flipud(np.rot90(img, 2))
    elif mode == 7:
        return np.flipud(np.rot90(img, 3))


# ===============================
# Dataset
# ===============================

class DatasetJPEG(data.Dataset):
    """
    JPEG artifact reduction dataset.
    Input images are DNG (RAW) files.
    """

    def __init__(self, opt):
        super().__init__()

        print('Dataset: JPEG artifact reduction from DNG (RAW) images')

        self.opt = opt
        self.patch_size = opt.get('H_size', 128)
        self.is_color = opt.get('is_color', True)

        # Training quality (single value)
        self.qf_train = [10, 20, 30, 40, 50, 60, 70]

        # Test qualities (single or list)
        self.qf_test =  [10, 20, 30, 40, 50, 60, 70] # opt.get('quality_factor_test', 40)


        root = opt['dataroot_H']
        self.paths_H = sorted(glob.glob(os.path.join(root, '*.dng')))
        if len(self.paths_H) == 0:
            raise RuntimeError(f'No DNG images found in {root}')

    def __len__(self):
        return len(self.paths_H)

    # ===============================
    # JPEG degradation (SAFE)
    # ===============================

    @staticmethod
    def jpeg_compress(img, q, is_color=True):
        """JPEG compression with guaranteed int quality."""
        q = int(q)
        if is_color:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(
                '.jpg', img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            img = cv2.imdecode(enc, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            _, enc = cv2.imencode(
                '.jpg', img,
                [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            img = cv2.imdecode(enc, 0)
        return img

    # ===============================
    # Main access
    # ===============================

    def __getitem__(self, index):
        H_path = self.paths_H[index]

        # Read RAW
        img_H = read_dng(H_path)
        img_L = img_H.copy()

        if self.opt['phase'] == 'train':

            H, W = img_H.shape[:2]
            ps_plus = self.patch_size + 8
            rnd_h = random.randint(0, max(0, H - ps_plus))
            rnd_w = random.randint(0, max(0, W - ps_plus))
            patch = img_H[rnd_h:rnd_h + ps_plus,
                        rnd_w:rnd_w + ps_plus]

            patch = augment_img(patch, random.randint(0, 7))

            img_H = patch.copy()
            img_L = patch.copy()

            # -------------------------------
            # RANDOM JPEG QUALITY (TRAIN)
            # -------------------------------
            q = random.choice(self.qf_train)

            if not self.is_color:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)
                img_H = img_L.copy()

            img_L = self.jpeg_compress(img_L, q, self.is_color)

            # Final crop
            H, W = img_H.shape[:2]
            rnd_h = random.randint(0, H - self.patch_size)
            rnd_w = random.randint(0, W - self.patch_size)

            img_H = img_H[rnd_h:rnd_h + self.patch_size,
                        rnd_w:rnd_w + self.patch_size]
            img_L = img_L[rnd_h:rnd_h + self.patch_size,
                        rnd_w:rnd_w + self.patch_size]

        else:
            # -------------------------------
            # VALID / TEST
            # -------------------------------
            q = random.choice(self.qf_test)

            # Ensure RGB
            if img_H.ndim == 2:
                img_H = cv2.cvtColor(img_H, cv2.COLOR_GRAY2RGB)

            # Center crop
            crop_h, crop_w = 256, 526
            H, W = img_H.shape[:2]
            sh = max((H - crop_h) // 2, 0)
            sw = max((W - crop_w) // 2, 0)
            img_H = img_H[sh:sh + crop_h, sw:sw + crop_w]
            img_L = img_H.copy()

            # JPEG degradation (TEST)
            if not self.is_color:
                img_H = cv2.cvtColor(img_H, cv2.COLOR_RGB2GRAY)
            img_L = self.jpeg_compress(img_L, q, self.is_color)

        # To tensor
        img_L = uint2tensor3(img_L)
        img_H = uint2tensor3(img_H)

        return {
            'L': img_L,
            'H': img_H,
            'L_path': H_path,
            'H_path': H_path
        }
