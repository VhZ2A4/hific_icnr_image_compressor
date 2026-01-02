# HiFiC-ICNR: Generative Image Compression with Refined Reconstruction

This repository implements a learned image compression framework based on **CompressAI**, enhanced with **HiFiC (High-Fidelity Compression)** training strategies. It features a custom **RRDB-based Refinement Module** and uses **ICNR Initialization** to achieve high-quality, artifact-free image reconstruction.

![Visual Comparison](./readme_comparison_optimized.jpg)

*"As shown in the picture `flowerbed_1_GAN_0.4019bpp_0.1045lpips`, this image represents the output of our GAN model on the flowerbed test sample. The model achieves a bit rate of roughly 0.4 bpp while maintaining high perceptual quality, evidenced by the low LPIPS score of 0.1045."*

## Key Features

* **Base Architecture**: Built upon the `mbt2018-mean` (Minnehan et al.) entropy model from [CompressAI](https://github.com/InterDigitalInc/CompressAI).
* **Advanced Refinement**: integrated a post-processing module using **RRDB** (Residual-in-Residual Dense Block) to recover high-frequency details.
* **Artifact Suppression**: Replaces standard upsampling with **PixelShufflePack** using **ICNR (Initialization to Convolution Nearest Neighbor Resize)** to eliminate checkerboard artifacts.
* **Generative Training**: Trained with a compound loss function including **MSE**, **LPIPS** (Perceptual Loss), and **Adversarial Loss** (PatchGAN Discriminator) for high perceptual quality.

## Project Structure

```text
.
├── main.py               # Training script (supports GAN training & multi-gpu prep)
├── compress_single.py    # Inference script for single image compression
├── requirements.txt      # Python dependencies
├── checkpoints/          # (Auto-generated) Saves model weights
├── data/                 # (Create this) Place your datasets here
├── test.png              # Pic required for the compression
└── results/              # (Auto-generated) Inference outputs
```

## Installation

It is recommended to use Conda to manage the environment.

### 1. Create Environment

Based on the provided configuration, Python 3.9 is recommended.

```bash
conda create -n jpeg_env python=3.9 -y
conda activate jpeg_env
```

### 2. Install Dependencies

Install the required packages, including PyTorch with CUDA 12.1 support2222.

```bash
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

> **Note**: Key dependencies include `compressai`, `lpips`, `torch`, `torchvision`, and `timm` (implied).

## Training

The training script (`main.py`) is configured to use **DIV2K** and **Flickr2K** datasets by default.

### 1. Prepare Data

Organize your dataset in the `data/` directory or modify the `Config.TRAIN_DIRS` path in `main.py`.

```python
# Default paths in main.py:
DATA_ROOT = "data"
TRAIN_DIRS = [
    os.path.join(DATA_ROOT, "DIV2K_train_HR"),
    os.path.join(DATA_ROOT, "Flickr2K", "Flickr2K_HR"), 
]
```

### 2. Run Training

Start the training process. The script handles Warmup (MSE only) followed by GAN training.

```bash
python main.py
```

**Training Hyperparameters:**

- **Batch Size**: 8

- **Patch Size**: 256x256

- **Total Epochs**: 100 (Warmup: 10)

- **Learning Rate**: 1e-4 (Cosine Annealing)

- **Lambda Weights**: Rate (1.0), MSE (100.0), LPIPS (5.0), GAN (0.05).

## Inference (Compression)

Use `compress_single.py` to compress a specific image and view the reconstructed result.

### 1. Setup

- Ensure you have a trained model checkpoint (e.g., `hific_icnr.pth`).

- Place your test image (e.g., `test.png`) in the project root.

### 2. Run Inference

```bash
python compress_single.py
```

### 3. Output

The script will display the **Bitrate (bpp)**, **Encoding Time**, and **Decoding Time**. The reconstructed image will be saved to the `results/` folder.

```textile
Original Size: 1920x1080
Bitrate:       0.2450 bpp
Encoding Time: 0.1200 s
Decoding Time: 0.1500 s
Reconstructed image saved to: results/output_0.245bpp.png
```

## Model Architecture Details

### Generator

- **Encoder/Decoder**: Standard hyperprior architecture.

- **Refinement**: A U-Net like structure containing `RRDB` blocks.

- **Upsampling**: `PixelShufflePack` class implements sub-pixel convolution with ICNR weight initialization.

### Discriminator

- **Multi-Scale Discriminator**: Uses 3 scales of `NLayerDiscriminator` (PatchGAN style) to capture texture details at different resolutions.

## License

This project is open-sourced under the MIT License.
