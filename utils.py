# utils.py
import torch
import math
import numpy as np
import datetime
import sys

from pytorch_msssim import ssim


# ==================================================================================
# Image Processing and Evaluation
# ==================================================================================

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr. Input image expected in range (0, 255).
    """
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor.")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).")

    # Assuming input image is in range (0, 255)
    image_01 = image / 255.
    r: torch.Tensor = image_01[..., 0, :, :]
    g: torch.Tensor = image_01[..., 1, :, :]
    b: torch.Tensor = image_01[..., 2, :, :]

    # BT.601 conversion
    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    # cb and cr are omitted as usually only Y channel is used for evaluation
    # cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    # cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    # Return only the Y channel (B, 1, H, W)
    return y.unsqueeze(-3)


def calc_psnr(sr, hr, rgb_range=255.0):
    """Calculate PSNR. Assumes inputs are aligned and cropped."""
    sr, hr = sr.double(), hr.double()
    
    max_val = float(rgb_range)
    
    # Clamp inputs
    sr = torch.clamp(sr, 0, max_val)
    hr = torch.clamp(hr, 0, max_val)

    if max_val > 1.0:
        sr = torch.round(sr)
        hr = torch.round(hr)

    diff = (sr - hr)
    # Detach to avoid autograd warning when converting to Python scalar
    mse_t = diff.pow(2).mean().detach()
    mse = float(mse_t.cpu().item())
    if mse < 1e-10:
        return 100.0
    # Standard PSNR formula: 10 * log10(MAX^2 / MSE)
    psnr = 10 * math.log10((max_val ** 2) / mse)
    return float(psnr)


def calc_ssim(sr, hr, rgb_range=255.0):
    """Calculate SSIM. Assumes inputs are aligned and cropped."""
    if ssim is None:
        print("Warning: SSIM calculation skipped as pytorch_msssim is not available.")
        return 0.0
        
    data_range = float(rgb_range)

    # Clamp inputs
    sr = torch.clamp(sr, 0, data_range)
    hr = torch.clamp(hr, 0, data_range)

    if data_range > 1.0:
        sr = torch.round(sr)
        hr = torch.round(hr)

    ssim_val = ssim(sr, hr, data_range=data_range, size_average=True)

    return float(ssim_val.detach().cpu().item())
    

# ==================================================================================
# Logging and Experiment Management
# ==================================================================================

def cur_timestamp_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m%d-%H%M")


class ExperimentLogger(object):
    """Redirects stdout to both terminal and a log file."""
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() # Flush immediately to ensure logs are saved
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_stat_dict(eval_sets=None):
    """
    Initializes the statistics dictionary dynamically based on the evaluation sets.
    """
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'l1_losses': [],
        'fft_losses': [],
        'intermediate_losses': [],
    }
    
    if eval_sets is None or len(eval_sets) == 0:
        return stat_dict

    for name in eval_sets:
        stat_dict[name] = {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {'value': 0.0, 'epoch': 0},
            'best_ssim': {'value': 0.0, 'epoch': 0}
        }
    return stat_dict