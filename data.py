# data.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import random
from typing import Tuple, List, Optional

# Disable albumentations online version check
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
try:
    import albumentations as A
except ImportError:
    print("Error: Albumentations not found. Please install it: pip install albumentations")
    exit(1)

def get_image_paths(dataroot):
    """Read image paths from the directory."""
    paths = []
    if dataroot is not None and os.path.exists(dataroot):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        for root, _, files in os.walk(dataroot):
            for name in files:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(root, name))
    return sorted(paths)

class TrainDataset(Dataset):
    """
    Training dataset for CS-SR. Loads HR images, performs random cropping and augmentation.
    """
    def __init__(self, dataroot, patch_size, colors=1, repeats=1, augment=True, rgb_range=1.0):
        self.paths = get_image_paths(dataroot)
        if not self.paths:
            print(f"Warning: No images found in {dataroot}")
        
        self.patch_size = patch_size
        self.colors = colors
        self.repeats = max(1, repeats)
        self.augment = augment
        self.rgb_range = rgb_range

        if self.augment:
            self.transform = A.Compose([
                A.RandomCrop(height=self.patch_size, width=self.patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=self.patch_size, width=self.patch_size)
            ])
            
    def __len__(self):
        return len(self.paths) * self.repeats

    def _get_image(self, idx):
        # Handle repeats
        img_path = self.paths[idx % len(self.paths)]
        
        # Read image (OpenCV reads in BGR format)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            # Fallback if image reading fails
            # print(f"Warning: Failed to read {img_path}. Using random fallback.")
            if len(self.paths) > 0:
                fallback_idx = random.randint(0, len(self.paths) - 1)
                img = cv2.imread(self.paths[fallback_idx], cv2.IMREAD_COLOR)
            if img is None:
                 img = np.random.randint(0, 256, (self.patch_size+50, self.patch_size+50, 3), dtype=np.uint8)
        
        return img

    def __getitem__(self, idx):
        img_hr_bgr = self._get_image(idx)
        H, W, _ = img_hr_bgr.shape

        # Ensure image is large enough before cropping
        if H < self.patch_size or W < self.patch_size:
            # Upscale if smaller than patch size using robust interpolation
            scale = max(self.patch_size / H, self.patch_size / W)
            img_hr_bgr = cv2.resize(img_hr_bgr, (int(W*scale)+1, int(H*scale)+1), interpolation=cv2.INTER_CUBIC)

        # Cropping and Augmentation
        try:
            augmented = self.transform(image=img_hr_bgr)
            img_hr = augmented['image']
        except Exception as e:
                # Final fallback: Resize
                print(f"Warning: Augmentation failed unexpectedly: {e}. Falling back to resize.")
                img_hr = cv2.resize(img_hr_bgr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # Convert color space
        if self.colors == 1:
            # Convert BGR to Y (Grayscale) - Using YCrCb Y channel is standard for SR/CS
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2YCrCb)[:,:,0]
            img_hr = np.expand_dims(img_hr, axis=-1)
        elif self.colors == 3:
            # Convert BGR to RGB
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor (HWC -> CHW)
        img_hr = img_hr.astype(np.float32) / 255.0 * self.rgb_range
        
        # HWC to CHW
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr.transpose((2, 0, 1)))).float()

        # CS-SR training only requires the HR image. 
        # We return a dummy LR tensor for compatibility with the (LR, HR) format.
        lr_dummy = torch.empty(0)
        return lr_dummy, img_hr


class TestDataset(Dataset):
    """
    Testing dataset for CS-SR.
    By default loads full HR images. If crop_hr_size is provided, performs random or center crop on HR.
    """
    def __init__(self, dataroot, colors=1, rgb_range=1.0,
                 crop_hr_size: Optional[int] = None,
                 random_crop: bool = True,
                 scale: int = 2,
                 pad_if_needed: bool = True,
                 supported_lr_resolutions: Optional[List[Tuple[int, int]]] = None):
        self.paths = get_image_paths(dataroot)
        if not self.paths:
            print(f"Warning: No images found in {dataroot}")
        self.colors = colors
        self.rgb_range = rgb_range
        self.crop_hr_size = crop_hr_size
        self.random_crop = random_crop
        self.scale = scale
        self.pad_if_needed = pad_if_needed
        # Normalize supported LR resolutions to list of tuples
        self.supported_lr_resolutions: List[Tuple[int, int]] = []
        if supported_lr_resolutions:
            for item in supported_lr_resolutions:
                try:
                    h_lr, w_lr = int(item[0]), int(item[1])
                    self.supported_lr_resolutions.append((h_lr, w_lr))
                except Exception:
                    continue

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        filename = os.path.basename(img_path)
        
        # Read image
        img_hr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img_hr is None:
             # Fallback
            print(f"Warning: Failed to read {img_path}. Using dummy image.")
            img_hr = np.zeros((128, 128, 3), dtype=np.uint8)

        # Optional crop on HR to ensure supported resolution
        H, W, _ = img_hr.shape
        # Decide target crop size: explicit -> auto from supported LR -> no crop
        do_crop = self.crop_hr_size is not None or (self.supported_lr_resolutions and self.random_crop is not None)
        if do_crop:
            if self.crop_hr_size is not None and self.crop_hr_size > 0:
                ch = cw = int(self.crop_hr_size)
            elif self.supported_lr_resolutions:
                # Choose the largest supported HR size that fits current image; else choose smallest and pad
                hr_sizes = [(h_lr*self.scale, w_lr*self.scale) for (h_lr, w_lr) in self.supported_lr_resolutions]
                # Sort by area descending
                hr_sizes.sort(key=lambda x: x[0]*x[1], reverse=True)
                chosen = None
                for (hh, ww) in hr_sizes:
                    if hh <= H and ww <= W:
                        chosen = (hh, ww)
                        break
                if chosen is None:
                    # Fallback to smallest; will pad if needed
                    chosen = hr_sizes[-1]
                ch, cw = int(chosen[0]), int(chosen[1])
            else:
                ch = cw = min(H, W)

            # Ensure HR crop size is divisible by scale
            if ch % self.scale != 0:
                ch = ch - (ch % self.scale)
            if cw % self.scale != 0:
                cw = cw - (cw % self.scale)
            # Guard
            ch = max(self.scale, ch)
            cw = max(self.scale, cw)
            if H < ch or W < cw:
                if self.pad_if_needed:
                    pad_h = max(0, ch - H)
                    pad_w = max(0, cw - W)
                    top = pad_h // 2
                    bottom = pad_h - top
                    left = pad_w // 2
                    right = pad_w - left
                    img_hr = cv2.copyMakeBorder(img_hr, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
                    H, W, _ = img_hr.shape
                else:
                    # fallback: resize to at least required crop size
                    scale_h = max(1.0, ch / max(1, H))
                    scale_w = max(1.0, cw / max(1, W))
                    s = max(scale_h, scale_w)
                    img_hr = cv2.resize(img_hr, (int(W*s)+1, int(H*s)+1), interpolation=cv2.INTER_CUBIC)
                    H, W, _ = img_hr.shape

            if self.random_crop:
                y0 = random.randint(0, H - ch)
                x0 = random.randint(0, W - cw)
            else:
                y0 = max(0, (H - ch) // 2)
                x0 = max(0, (W - cw) // 2)
            img_hr = img_hr[y0:y0+ch, x0:x0+cw, :]

        # Convert color space
        if self.colors == 1:
            # Convert BGR to Y (Grayscale)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2YCrCb)[:,:,0]
            img_hr = np.expand_dims(img_hr, axis=-1)
        elif self.colors == 3:
            # Convert BGR to RGB
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor (HWC -> CHW)
        img_hr = img_hr.astype(np.float32) / 255.0 * self.rgb_range
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr.transpose((2, 0, 1)))).float()

        # Return dummy LR, HR, filename
        lr_dummy = torch.empty(0)
        return lr_dummy, img_hr, filename

# ==================================================================================
# Main entry point for creating dataloaders (Interface for train.py/test.py)
# ==================================================================================

def create_datasets(args):
    """
    Creates training and validation dataloaders based on configuration arguments.
    """
    
    # 1. Training Dataset
    train_ds = TrainDataset(
        dataroot=os.path.join(args.data_path, args.training_dataset),
        patch_size=args.patch_size,
        colors=args.colors,
        repeats=args.data_repeat,
        augment=bool(args.data_augment),
        rgb_range=args.rgb_range
    )
    
    # Use deterministic workers if possible
    g = torch.Generator()
    g.manual_seed(2024)
    
    def _seed_worker(worker_id):
        # Disable OpenCV multithreading in workers to avoid CPU oversubscription/contention
        cv2.setNumThreads(0)
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=True if args.threads > 0 else False,
        prefetch_factor=4 if args.threads > 0 else None
    )

    # 2. Validation Datasets
    valid_loaders = []
    # Test/Validation cropping options
    test_hr_crop = bool(getattr(args, 'test_hr_crop', True))
    # Determine supported LR and corresponding HR sizes (from top-level config)
    supported_lr_res = []
    try:
        supported_lr_res = list(getattr(args, 'supported_lr_resolutions', []))
    except Exception:
        supported_lr_res = []
    hr_sizes = [(int(h)*args.scale, int(w)*args.scale) for (h, w) in supported_lr_res] if supported_lr_res else []

    # None means: auto-derive; prefer training patch size if valid, else choose the smallest supported HR size
    raw_crop_size = getattr(args, 'test_hr_crop_size', None)
    if raw_crop_size is not None and str(raw_crop_size).lower() != 'none':
        test_hr_crop_size = int(raw_crop_size)
    else:
        preferred = int(getattr(args, 'patch_size', 0))
        if preferred and (preferred % args.scale == 0):
            test_hr_crop_size = preferred
        elif hr_sizes:
            # choose smallest supported HR size (area) to align with training stability
            test_hr_crop_size = int(sorted(hr_sizes, key=lambda x: x[0]*x[1])[0][0])
        else:
            test_hr_crop_size = int(getattr(args, 'patch_size', 128))
    test_random_crop = bool(getattr(args, 'test_random_crop', True))
    test_pad_if_needed = bool(getattr(args, 'test_pad_if_needed', True))

    for dataset_name in args.eval_sets:
        val_ds = TestDataset(
            dataroot=os.path.join(args.data_path, dataset_name),
            colors=args.colors,
            rgb_range=args.rgb_range,
            crop_hr_size=test_hr_crop_size,
            random_crop=test_random_crop,
            scale=args.scale,
            pad_if_needed=test_pad_if_needed,
            supported_lr_resolutions=supported_lr_res
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1, # Validation typically uses batch size 1
            shuffle=False,
            num_workers=args.threads,
            pin_memory=True
        )
        valid_loaders.append({'name': dataset_name, 'dataloader': val_loader})

    return train_loader, valid_loaders
