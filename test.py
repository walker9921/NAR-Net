# test.py
import torch
import torch.nn.functional as F
import argparse, yaml
import utils
import os
import time
import numpy as np
from tqdm import tqdm
import sys
from torchvision.utils import save_image
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from model.NAR_Net import NAR_Net

parser = argparse.ArgumentParser(description='NAR_Net Testing')
parser.add_argument('--config', type=str, required=True, help = 'config file used for training (defines model architecture)')
parser.add_argument('--checkpoint', type=str, required=True, help = 'path to the trained model checkpoint (.pt file)')
parser.add_argument('--save_dir', type=str, default='./testing_results', help = 'directory to save the results')
parser.add_argument('--gpu_ids', type=str, default="[0]", help='GPU ids for testing (e.g., "[0]")')


# ==================================================================================
# Dataset Loading (test only — creates validation dataloaders without train data)
# ==================================================================================

from data import TestDataset

def create_test_dataloaders(args):
    """Create validation dataloaders only (no training dataset needed)."""
    from torch.utils.data import DataLoader
    valid_loaders = []
    for dataset_name in args.eval_sets:
        val_ds = TestDataset(
            dataroot=os.path.join(args.data_path, dataset_name),
            colors=args.colors,
            rgb_range=args.rgb_range,
            crop_hr_size=None,
            random_crop=False,
            scale=args.scale,
            pad_if_needed=True,
            supported_lr_resolutions=[]
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.threads, pin_memory=True)
        valid_loaders.append({'name': dataset_name, 'dataloader': val_loader})
    return valid_loaders

# ==================================================================================


if __name__ == '__main__':

    args = parser.parse_args()
    
    # Load configuration from YAML to define the model architecture
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       
       # Store command line arguments that should override config
       checkpoint_cmd = opt['checkpoint']
       save_dir_cmd = opt['save_dir']
       gpu_ids_cmd = opt['gpu_ids']

       # Update args with YAML content
       opt.update(yaml_args)
       
       # Re-apply command line overrides
       opt['checkpoint'] = checkpoint_cmd
       opt['save_dir'] = save_dir_cmd
       opt['gpu_ids'] = eval(gpu_ids_cmd) # Parse GPU IDs string "[0]" to list [0]


    ## set visible gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    
    print(f'## Testing {args.model_name} | X{args.scale} ##')

    # Set seed
    torch.manual_seed(2024)

    ## select active gpu devices
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('## use cuda & cudnn for acceleration! ##')
        device = torch.device('cuda')
    else:
        print('## use cpu! ##')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    # AMP disabled for inference to ensure fp32 consistency with reference results
    print("## AMP (inference): disabled (fp32) ##")

    ## create dataset for validating (test-only, no training data required)
    valid_dataloaders = create_test_dataloaders(args)

    ## definitions of model
    # 准备模型初始化参数 (适应新的YAML结构)
    if not hasattr(args, 'sensing') or not hasattr(args, 'prior'):
        raise ValueError("Configuration file must contain 'sensing' and 'prior' sections.")
    sensing_config = args.sensing
    prior_config = args.prior.copy()
    prior_config['color_channel'] = args.colors

    sensing_type_raw = sensing_config.get('type', 'AKCS')
    sensing_type = str(sensing_type_raw).strip().lower()
    if sensing_type in ('nkcs', 'nso'):
        sensing_mode = 'nkcs'
    elif sensing_type == 'kcs':
        sensing_mode = 'standard'
    elif sensing_type == 'akcs':
        sensing_mode = 'asymmetric'
    else:
        raise ValueError("sensing.type must be either 'KCS', 'AKCS' or 'NKCS'.")
    
    default_multi_crs = [0.01, 0.04, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    multi_cr_list = sorted(getattr(args, 'multi_cr_list', default_multi_crs))
    TEACHER_CR = float(max(multi_cr_list))

    model_args = {
        'stages': args.stages,
        'scale_factor': args.scale,
        'max_cr': TEACHER_CR,
        'color_channel': args.colors,
        'sensing_mode': sensing_mode,
        'binary_sensing': sensing_config['binary'],
        'normalize_sensing': sensing_config['normalize'],
        'binarization_mode': sensing_config['binarization_mode'],
        'supported_lr_resolutions': getattr(args, 'supported_lr_resolutions', None),
        'prior_config': prior_config,
    }

    try:
        model = NAR_Net(**model_args)
    except Exception as e:
        raise ValueError(f'Error creating model: {e}')

    # 与当前训练脚本保持一致：单卡模型，不再包一层 DataParallel，
    # 以便与无 "module." 前缀的 checkpoint state_dict 对齐。
    model = model.to(device)

    # Initialize LPIPS metric (grayscale will be replicated to 3 channels when computing)
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    lpips_model.eval()

    ## load checkpoint
    print('## loading checkpoint: {}! ##'.format(args.checkpoint))  
    # Robust checkpoint loading across PyTorch versions and saved formats.
    # Prefer weights_only=True; if it fails, allowlist known scheduler classes; if still failing, fallback to unsafe load.
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception as e:
        print(f"[WARN] Safe load (weights_only=True) failed: {e}")
        # Try allowlisting common scheduler classes used in this repo
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            import torch.optim.lr_scheduler as _lrs
            try:
                from scheduler import GradualWarmupScheduler as _GWS  # local scheduler
                add_safe_globals([_lrs.CosineAnnealingLR, _GWS])
            except Exception:
                add_safe_globals([_lrs.CosineAnnealingLR])
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
            print("[INFO] Loaded with allowlisted safe globals.")
        except Exception as e2:
            print(f"[WARN] Allowlisting failed or unavailable: {e2}. Falling back to unsafe load (weights_only=False).")
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', 'N/A')
        print(f"## Loaded model from epoch {epoch} ##")
    except KeyError:
        model.load_state_dict(ckpt) # Handle direct state_dict save
        print("## Loaded model weights (Epoch info not found) ##")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Get model reference (handle DataParallel) for checking attributes
    model_ref = model.module if hasattr(model, 'module') else model

    ## create folder for test results
    save_path_base = os.path.join(args.save_dir, f"{args.model_name}_x{args.scale}_MultiCR")
    os.makedirs(save_path_base, exist_ok=True)

    # ## testing
    test_log = ''
    print(f"\nStarting Evaluation (Simulation Mode)... Results saving to {save_path_base}")
    
    # --- Helper: Tiled forward (256x256) with padding & stitch back ---
    def tiled_forward_full_image(model, hr_img: torch.Tensor, cr: float, tile_size: int = 256):
        """
        Returns:
            sr_full: Reconstructed full image [B, C, H, W]
        """
        # hr_img: [B(=1), C, H, W]
        B, C, H, W = hr_img.shape
        assert B == 1, "Testing expects batch size 1 per image."
        # Compute padded size to multiples of tile_size
        H_pad = ((H + tile_size - 1) // tile_size) * tile_size
        W_pad = ((W + tile_size - 1) // tile_size) * tile_size
        pad_b = H_pad - H
        pad_r = W_pad - W
        if pad_b > 0 or pad_r > 0:
            # Pad order: (left, right, top, bottom)
            pad_mode = 'reflect' if H > 1 and W > 1 else 'replicate'
            hr_padded = F.pad(hr_img, (0, pad_r, 0, pad_b), mode=pad_mode)
        else:
            hr_padded = hr_img

        # Unfold into non-overlapping tiles
        nH = H_pad // tile_size
        nW = W_pad // tile_size
        tiles = hr_padded.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
        # Shape: [B, C, nH, nW, tile, tile] -> [B*nH*nW, C, tile, tile]
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * nH * nW, C, tile_size, tile_size)

        # Forward all tiles as one batch (AMP applied by caller)
        sr_tiles, _ = model(GT_HR=tiles, cr=cr)

        # Fold back to full image
        sr_tiles = sr_tiles.view(B, nH, nW, C, tile_size, tile_size).permute(0, 3, 1, 4, 2, 5).contiguous()
        sr_padded = sr_tiles.view(B, C, H_pad, W_pad)
        # Crop to original size
        sr_full = sr_padded[:, :, :H, :W]
        
        return sr_full

    with torch.no_grad():
        for test_cr in multi_cr_list:
            print(f"\n## Testing CR: {test_cr} ##")
            cr_str = f"CR{test_cr:.2f}".replace('.', 'p')
            
            for valid_dataloader_dict in valid_dataloaders:
                avg_psnr, avg_ssim = 0.0, 0.0
                avg_lpips = 0.0
                lpips_count = 0
                total_inference_time = 0.0
                name = valid_dataloader_dict['name']
                loader = valid_dataloader_dict['dataloader']
                processed_count = 0
                
                save_path_dataset = os.path.join(save_path_base, cr_str, name)
                os.makedirs(save_path_dataset, exist_ok=True)

                for batch in tqdm(loader, desc=f"Testing {name} (CR={test_cr})", ncols=100):
                    
                    # Data Loading: Assume format (LR_dummy, HR, img_name)
                    if len(batch) >= 3:
                        hr = batch[1]
                        img_name = batch[2]
                    else:
                        tqdm.write(f"Warning: Unexpected batch format in test loader {name}. Skipping.")
                        continue

                    hr = hr.to(device)
                    
                    # --- Dimension Check and Support Verification ---
                    H_hr, W_hr = hr.shape[2], hr.shape[3]
                    
                    # 检查全图尺寸是否被支持
                    full_lr_h = H_hr // args.scale
                    full_lr_w = W_hr // args.scale
                    full_key = f"{full_lr_h}x{full_lr_w}"
                    
                    # 获取默认的 tile size
                    default_tile_sz = int(getattr(args, 'patch_size', 128))
                    default_tile_lr = default_tile_sz // args.scale
                    default_tile_key = f"{default_tile_lr}x{default_tile_lr}"

                    # 决策逻辑：优先全图，其次分块
                    if full_key in model_ref.sensing_ops: # type: ignore
                        # 模型支持全图，直接使用全图尺寸作为 tile_size (即不分块)
                        actual_tile_sz = H_hr 
                    elif default_tile_key in model_ref.sensing_ops: # type: ignore
                        # 模型不支持全图，退回到默认分块
                        actual_tile_sz = default_tile_sz
                    else:
                        tqdm.write(f"WARNING: Skipping {img_name[0]}: neither full size {full_key} nor patch size {default_tile_key} supported.")
                        continue
                    
                    processed_count += 1

                    # Forward pass (fp32, no AMP)
                    inf_start = time.perf_counter()
                    with torch.no_grad():
                        # 使用决策后的 actual_tile_sz
                        sr_full = tiled_forward_full_image(model, hr, cr=test_cr, tile_size=actual_tile_sz)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    total_inference_time += time.perf_counter() - inf_start
                    
                    # --- Evaluation (PSNR/SSIM using skimage on [0,255] float, no edge cropping) ---
                    # Convert to numpy for skimage metrics (consistent with test_narnet_continuous_cr.py)
                    sr_np = sr_full.squeeze().clamp(0, 1).cpu().numpy()  # [H, W] for grayscale
                    hr_np = hr.squeeze().clamp(0, 1).cpu().numpy()       # [H, W] for grayscale

                    # skimage metrics on [0, 255] range (float, no rounding)
                    psnr = compare_psnr(hr_np * 255, sr_np * 255, data_range=255)
                    ssim = compare_ssim(hr_np * 255, sr_np * 255, data_range=255)
                    avg_psnr += psnr
                    avg_ssim += ssim

                    # LPIPS (replicate grayscale to RGB)
                    lpips_this = None
                    sr_lp = sr_full
                    hr_lp = hr
                    # Normalize to [0,1]
                    if args.rgb_range != 1.0:
                        sr_lp = sr_lp / args.rgb_range
                        hr_lp = hr_lp / args.rgb_range
                    # Ensure 3 channels for LPIPS
                    if sr_lp.shape[1] == 1:
                        sr_lp = sr_lp.repeat(1, 3, 1, 1)
                    if hr_lp.shape[1] == 1:
                        hr_lp = hr_lp.repeat(1, 3, 1, 1)
                    # LPIPS expects inputs in [-1, 1]
                    sr_lp_norm = sr_lp * 2.0 - 1.0
                    hr_lp_norm = hr_lp * 2.0 - 1.0
                    lp_val = float(lpips_model(sr_lp_norm, hr_lp_norm).mean().item())
                    avg_lpips += lp_val
                    lpips_this = lp_val
                    lpips_count += 1

                    # Save concatenated GT|Recon with metrics in filename
                    if args.save_image:
                        # Prepare visualization tensors in [0,1]
                        sr_vis = torch.clamp(sr_full[0], 0, args.rgb_range)
                        hr_vis = torch.clamp(hr[0], 0, args.rgb_range)
                        if args.rgb_range != 1.0:
                            sr_vis = sr_vis / args.rgb_range
                            hr_vis = hr_vis / args.rgb_range
                        # Concatenate along width (CHW, dim=2)
                        concat_vis = torch.cat([hr_vis, sr_vis], dim=2)
                        base, ext = os.path.splitext(img_name[0])
                        if lpips_this is not None:
                            out_name = f"{base}_PSNR{psnr:.2f}_SSIM{ssim:.4f}_LPIPS{lpips_this:.4f}.png"
                        else:
                            out_name = f"{base}_PSNR{psnr:.2f}_SSIM{ssim:.4f}.png"
                        save_image(concat_vis, os.path.join(save_path_dataset, out_name))

                if processed_count > 0:
                    avg_psnr = round(avg_psnr/processed_count, 4)
                    avg_ssim = round(avg_ssim/processed_count, 6)
                    avg_infer_time_ms = (total_inference_time/processed_count) * 1000.0
                    if lpips_count > 0:
                        avg_lpips_val = avg_lpips / lpips_count
                        test_log += (
                            f"[{cr_str}] {name} dataset: avg_psnr: {avg_psnr:.4f}, avg_ssim: {avg_ssim:.6f}, "
                            f"avg_lpips: {avg_lpips_val:.6f}, avg_infer_time: {avg_infer_time_ms:.2f} ms "
                            f"(Processed {processed_count} images).\n"
                        )
                    else:
                        test_log += (
                            f"[{cr_str}] {name} dataset: avg_psnr: {avg_psnr:.4f}, avg_ssim: {avg_ssim:.6f}, "
                            f"avg_infer_time: {avg_infer_time_ms:.2f} ms (Processed {processed_count} images).\n"
                        )
                else:
                    test_log += f"[{cr_str}] WARNING: No images processed for dataset {name}.\n"


    # print log & flush out
    print("\n--- Final Results ---")
    print(test_log)
    
    # Save log file
    with open(os.path.join(save_path_base, 'test_log.txt'), 'w') as f:
        f.write(test_log)
        
    print(f"Results saved in {save_path_base}")