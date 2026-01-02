import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from compressai.zoo import mbt2018_mean

from models import BalancedHiFiCGenerator

# ==========================================
# B. Main Logic
# ==========================================

def pad_image(img, factor=64):
    _, _, h, w = img.size()
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    return F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

def run_inference():
    # --- Configuration ---
    MODEL_PATH = "hific_icnr.pth" 
    INPUT_IMG = "test.png"                             
    OUTPUT_DIR = "results"                             
    QUALITY_LEVEL = 3
    # ---------------------

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"Device: {DEVICE}")
    print("Initializing model...")
    net = BalancedHiFiCGenerator(quality=QUALITY_LEVEL).to(DEVICE)
    
    # 1. Smart Weight Loading
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'g_state_dict' in checkpoint:
        print("Detected training checkpoint format. Loading 'g_state_dict'...")
        state_dict = checkpoint['g_state_dict']
    else:
        state_dict = checkpoint

    # Handle weight loading (filter mismatched keys)
    model_dict = net.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Ignoring shape mismatch: {k} (ckpt:{v.shape} vs model:{model_dict[k].shape})")
        else:
            # Handle potential 'module.' prefix from DataParallel
            k_clean = k.replace('module.', '')
            if k_clean in model_dict and v.shape == model_dict[k_clean].shape:
                new_state_dict[k_clean] = v
            else:
                pass 

    net.load_state_dict(new_state_dict, strict=False)
    
    # 2. Regenerate Entropy Model Tables (CDF Tables)
    print("Updating entropy bottleneck (crucial for correct decoding)...")
    net.backbone.update(force=True)
    net.eval()

    # 3. Load Image
    if not os.path.exists(INPUT_IMG):
        print(f"Error: Input image not found at {INPUT_IMG}")
        return

    img_pil = Image.open(INPUT_IMG).convert('RGB')
    x = transforms.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)
    orig_h, orig_w = x.shape[2], x.shape[3]
    
    # Padding
    x_padded = pad_image(x)

    # 4. Compress
    print(f"Compressing image ({orig_w}x{orig_h})...")
    t0 = time.time()
    with torch.no_grad():
        out_enc = net.backbone.compress(x_padded)
    enc_time = time.time() - t0

    # Calculate Bitrate
    total_bits = sum(len(s)*8 for s_list in out_enc['strings'] for s in s_list)
    bpp = total_bits / (orig_h * orig_w)

    # 5. Decompress + Refine
    print(f"Decompressing (BPP: {bpp:.4f})...")
    t1 = time.time()
    with torch.no_grad():
        # Backbone decoding
        out_dec = net.backbone.decompress(out_enc['strings'], out_enc['shape'])
        
        # Refine module enhancement
        x_hat = net.refine(out_dec['x_hat'])
        
        # Crop to original size and clamp values
        x_hat = x_hat[:, :, :orig_h, :orig_w]
        x_hat = x_hat.clamp(0, 1)
    dec_time = time.time() - t1

    # 6. Output Results
    print("-" * 40)
    print(f"Original Size: {orig_w}x{orig_h}")
    print(f"Bitrate:       {bpp:.4f} bpp")
    print(f"Encoding Time: {enc_time:.4f} s")
    print(f"Decoding Time: {dec_time:.4f} s")
    print("-" * 40)
    
    save_path = os.path.join(OUTPUT_DIR, f"output_{bpp:.3f}bpp.png")
    save_image(x_hat, save_path)
    print(f"Reconstructed image saved to: {save_path}")

if __name__ == "__main__":
    run_inference()