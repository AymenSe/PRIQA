import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

# Ensure these imports work in your environment
from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# Task: 'jpeg_car' (grayscale) or 'color_jpeg_car' (color)
TASK = 'color_jpeg_car'
# Input folder containing your compressed images
FOLDER_LQ = 'test_images/Dist' 
# Folder containing Ground Truth (Ref) images. Set to None if you don't have them.
FOLDER_GT = 'test_images' # Example: 'models/test_images' or None

quality_to_weight = {
    10: 'model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth',
    20: 'model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth',
    30: 'model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth',
    40: 'model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth'
}

# ==============================================================================
# CLASSES AND FUNCTIONS
# ==============================================================================

class Config:
    """Helper class to convert variables to an argument-like object"""
    def __init__(self, task, scale, noise, jpeg, training_patch_size, large_model, model_path, folder_lq, folder_gt, tile, tile_overlap):
        self.task = task
        self.scale = scale
        self.noise = noise
        self.jpeg = jpeg
        self.training_patch_size = training_patch_size
        self.large_model = large_model
        self.model_path = model_path
        self.folder_lq = folder_lq
        self.folder_gt = folder_gt
        self.tile = tile
        self.tile_overlap = tile_overlap

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 grayscale JPEG compression artifact reduction
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 color JPEG compression artifact reduction
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_lq  # Use Low Quality folder for input
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    
    # Example: path = "test_images/I70_10_01.jpg"
    # We want to find the corresponding Ground Truth
    
    img_gt = None
    img_lq = None

    if args.task in ['color_dn']:
        # Denoising logic...
        if args.folder_gt:
            gt_path = os.path.join(args.folder_gt, imgname + imgext) # Adjust based on naming convention
            img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        
        if img_gt is not None:
            np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # 006 color JPEG compression artifact reduction
    elif args.task in ['color_jpeg_car', 'jpeg_car']:
        # Load LQ Image
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_lq is None:
            print(f"Error reading LQ image: {path}")
            return imgname, None, None
            
        img_lq = img_lq.astype(np.float32)/ 255.

        # Load GT Image (Only if FOLDER_GT is provided)
        if args.folder_gt is not None:
            # Assumes the filename structure matches. 
            # If your GT files have different names (e.g. "I70.png" vs "I70_10_01.jpg"), modify logic here:
            base_name = imgname.split('_')[0] # Example: I70 from I70_10_01
            gt_path = os.path.join(args.folder_gt, f'{base_name}.png') 
            
            # Check if file exists before reading
            if os.path.exists(gt_path):
                img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                if img_gt is not None:
                    img_gt = img_gt.astype(np.float32) / 255.
            else:
                # Use this else block if you want to know which GTs are missing
                pass

    return imgname, img_lq, img_gt

def test(img_lq, model, args, window_size):
    if args.tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def process_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    print(f'Processing Quality: {args.jpeg} | Model: {args.model_path}')

    # Download model if it doesn't exist
    if os.path.exists(args.model_path):
        print(f'Loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        print(f'Downloading model to {args.model_path}...')
        try:
            r = requests.get(url, allow_redirects=True)
            open(args.model_path, 'wb').write(r.content)
        except Exception as e:
            print(f"Failed to download model: {e}")
            return

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # Setup paths
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Reading images from: {folder}')
    print(f'Saving results to: {save_dir}')

    # Check if images exist
    images = sorted(glob.glob(os.path.join(folder, '*')))
    if not images:
        print(f"\n[WARNING] No images found in '{folder}'!")
        print("Please check your FOLDER_LQ path in the configuration.")
        return

    # Metrics
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []

    # Processing loop
    for idx, path in enumerate(images):
        # Read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  
        
        if img_lq is None:
            continue

        # Preprocessing
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # Inference
        with torch.no_grad():
            # Pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = test(img_lq, model, args, window_size)
            
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # Postprocessing and Saving
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        save_path = f'{save_dir}/{imgname}_SwinIR.png'
        cv2.imwrite(save_path, output)

        # Evaluate metrics (Only if Ground Truth exists)
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car', 'color_jpeg_car']:
                psnrb = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=False)
                test_results['psnrb'].append(psnrb)
                if args.task in ['color_jpeg_car']:
                    psnrb_y = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)
                    test_results['psnrb_y'].append(psnrb_y)
            print(f'Processed {idx}: {imgname} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}')
        else:
            print(f'Processed {idx}: {imgname} -> Saved to {save_path}')

    # Summary
    if len(test_results['psnr']) > 0:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(f'-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
    print('Finished batch.\n')


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Settings that remain constant
    SCALE = 1
    NOISE_LEVEL = 15          
    TRAINING_PATCH_SIZE = 128
    LARGE_MODEL = False       
    TILE = None               
    TILE_OVERLAP = 32

    # Loop through the qualities
    for quality, weight_path in quality_to_weight.items():
        print(f"==================================================")
        print(f"STARTING TASK FOR JPEG QUALITY: {quality}")
        print(f"==================================================")

        # Create Args for this specific iteration
        args = Config(
            task=TASK,
            scale=SCALE,
            noise=NOISE_LEVEL,
            jpeg=quality,
            training_patch_size=TRAINING_PATCH_SIZE,
            large_model=LARGE_MODEL,
            model_path=weight_path,
            folder_lq=FOLDER_LQ,
            folder_gt=FOLDER_GT,
            tile=TILE,
            tile_overlap=TILE_OVERLAP
        )

        process_pipeline(args)