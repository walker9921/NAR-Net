# NAR-Net

Official implementation of NAR-Net: a physics-guided deep unfolding network for Single-Pixel Imaging (SPI) compressive sensing reconstruction with continuous compression ratios.

## Quick Start

### Environment

```bash
pip install torch torchvision numpy opencv-python pyyaml tqdm scikit-image lpips einops albumentations
```

### Download Pretrained Model & Datasets

Our official models and testsets can be downloaded on [Baidu Netdisk](https://pan.baidu.com/s/1U_k3VgjKlkYgxlpsCr-GEg?pwd=p3s4) (code: `p3s4`).

Please place them in the corresponding directories:

- `pretrain_model/model_latest.pt`
- `datasets/` (BSD68, McM, Set11)

### Test

```bash
python test.py --config configs/config_nso.yaml --checkpoint pretrain_model/model_latest.pt
```

## Results (PSNR / SSIM)

| CR | Set11 | BSD68 | McM |
|---|---|---|---|
| 0.01 | 24.41 / 0.707 | 23.58 / 0.576 | 25.52 / 0.687 |
| 0.04 | 29.81 / 0.879 | 26.67 / 0.732 | 30.33 / 0.846 |
| 0.10 | 33.87 / 0.939 | 29.48 / 0.839 | 34.23 / 0.923 |
| 0.25 | 38.59 / 0.972 | 33.83 / 0.931 | 39.40 / 0.971 |
| 0.50 | 43.32 / 0.987 | 39.58 / 0.978 | 45.00 / 0.990 |
