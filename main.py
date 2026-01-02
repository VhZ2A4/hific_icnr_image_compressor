import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models import BalancedHiFiCGenerator, MultiScaleDiscriminator

# ==========================================
# 1. Config
# ==========================================
class Config:
    # Path Configuration
    DATA_ROOT = "data"
    TRAIN_DIRS = [
        os.path.join(DATA_ROOT, "DIV2K_train_HR"),
        os.path.join(DATA_ROOT, "Flickr2K", "Flickr2K_HR"), 
    ]
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    NUM_WORKERS = 8       
    PERSISTENT_WORKERS = True 
    PREFETCH_FACTOR = 4   
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 8
    PATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    WARMUP_EPOCHS = 10
    
    # --- Loss Weights ---
    LAMBDA_RATE = 1.0
    LAMBDA_MSE = 100.0
    LAMBDA_LPIPS = 5.0
    LAMBDA_GAN = 0.05
    LAMBDA_FM = 10.0

# ==========================================
# 2. Dataset
# ==========================================
def calc_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else 100

class ImageFolderDataset(Dataset):
    def __init__(self, folders, patch_size=256):
        self.image_paths = []
        if isinstance(folders, str): folders = [folders]
        for folder in folders:
            self.image_paths.extend(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
            self.image_paths.extend(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
            self.image_paths.extend(glob.glob(os.path.join(folder, "**", "*.jpeg"), recursive=True))
        
        print(f"Dataset loaded: {len(self.image_paths)} images found.")
        
        self.transform = transforms.Compose([
            transforms.Resize(patch_size),
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# ==========================================
# 3. Core Architecture (ICNR + PixelShuffle)
# ==========================================

# --- Basic RRDB Components ---
class MakeDense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
    def forward(self, x):
        out = F.leaky_relu(self.conv(x), 0.2, True)
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        modules = []
        for i in range(nDenselayer):
            modules.append(MakeDense(nChannels + i * growthRate, growthRate))
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels + nDenselayer * growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        return out + x * 0.2

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(nf, 3, gc)
        self.RDB2 = RDB(nf, 3, gc)
        self.RDB3 = RDB(nf, 3, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

# --- PixelShufflePack with ICNR Initialization ---
class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, upscale_kernel=3):
        super(PixelShufflePack, self).__init__()
        self.scale_factor = scale_factor
        
        # For PixelShuffle, channels need to be expanded by scale_factor^2
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 
                              kernel_size=upscale_kernel, padding=upscale_kernel//2)
        
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Execute ICNR initialization (Critical step)
        self.icnr_init(self.conv.weight, scale_factor=scale_factor)
        
    def icnr_init(self, tensor, scale_factor=2, init=nn.init.kaiming_normal_):
        out_channels, in_channels, height, width = tensor.size()
        transpose_in_channels = out_channels // (scale_factor ** 2)
        
        kernel_shape = (transpose_in_channels, in_channels, height, width)
        kernel = torch.zeros(kernel_shape, dtype=tensor.dtype, device=tensor.device)
        kernel = init(kernel) # Base initialization
        
        # Expand kernel to PixelShuffle dimensions
        kernel = kernel.contiguous().view(transpose_in_channels, in_channels, -1)
        kernel = kernel.repeat(scale_factor ** 2, 1, 1) # Repeat across channel dimension
        kernel = kernel.contiguous().view(out_channels, in_channels, height, width)
        
        with torch.no_grad():
            tensor.copy_(kernel)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

# --- Refinement Module ---
class AdvancedRefinementModule(nn.Module):
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        
        # 1. Encoder
        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, nf, 3, 1, 0),
            nn.LeakyReLU(0.2, True)
        )
        
        # 2. Downsample
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf*2, 3, 2, 0), 
            nn.LeakyReLU(0.2, True)
        )
        
        # 3. Body
        self.body = nn.Sequential(
            nn.RRDB(nf*2),
            nn.RRDB(nf*2),
            nn.RRDB(nf*2)
        )
        
        # 4. Upsample
        self.up = nn.Sequential(
            PixelShufflePack(nf*2, nf, scale_factor=2), 
            nn.LeakyReLU(0.2, True)
        )
        
        # 5. Output
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1), 
            nn.Conv2d(nf, in_ch, 3, 1, 0) 
        )
        
        self.skip_conv = nn.Conv2d(nf, nf, 1, 1, 0)

    def forward(self, x):
        f1 = self.conv_in(x)
        f2 = self.down(f1)
        f3 = self.body(f2)
        f4 = self.up(f3) 
        res = self.conv_out(f4 + self.skip_conv(f1))
        return x + res

# ==========================================
# 4. Model Integration
# ==========================================

class BalancedHiFiCGenerator(nn.Module):
    def __init__(self, quality=3):
        super().__init__()
        print(f"Loading backbone: mbt2018-mean (quality={quality})...")
        self.backbone = mbt2018_mean(quality=quality, pretrained=True) 
        self.refine = AdvancedRefinementModule(in_ch=3, nf=64)
    
    def forward(self, x):
        out = self.backbone(x)
        out['x_hat'] = self.refine(out['x_hat'])
        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
        ))
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)),
            nn.LeakyReLU(0.2, True)
        ))
        self.layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))

    def forward(self, x):
        features = []
        out = x
        for layer in self.layers:
            out = layer(out)
            features.append(out)
        return out, features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_D=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(NLayerDiscriminator(input_nc, ndf, n_layers))
            
    def forward(self, x):
        result = []
        feats = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            out, feat = d(x)
            result.append(out)
            feats.append(feat)
        return result, feats

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    print(f"Hardware: {Config.DEVICE}")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    dataset = ImageFolderDataset(Config.TRAIN_DIRS, patch_size=Config.PATCH_SIZE)
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,    # Parallel loading
        pin_memory=True,                   # Pin memory to accelerate CPU->GPU transfer
        prefetch_factor=Config.PREFETCH_FACTOR, # Prefetch batches to fill pipeline
        persistent_workers=Config.PERSISTENT_WORKERS # Avoid process destruction/recreation
    )
    
    # Models
    net_g = BalancedHiFiCGenerator().to(Config.DEVICE)
    net_d = MultiScaleDiscriminator(num_D=3).to(Config.DEVICE)
    
    # Losses
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss() 
    print("Loading LPIPS (AlexNet)...")
    # Make sure 'lpips' library is imported
    lpips_criterion = lpips.LPIPS(net='alex').to(Config.DEVICE).eval()
    
    # Optimizers
    optimizer_g = optim.Adam(net_g.parameters(), lr=Config.LEARNING_RATE)
    optimizer_d = optim.Adam(net_d.parameters(), lr=Config.LEARNING_RATE)
    
    aux_parameters = [p for n, p in net_g.backbone.named_parameters() if n.endswith(".quantiles")]
    optimizer_aux = optim.Adam(aux_parameters, lr=1e-3)

    # Cosine Annealing Scheduler:
    scheduler_steps = Config.EPOCHS - Config.WARMUP_EPOCHS
    
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, 
        T_max=scheduler_steps,
        eta_min=1e-7
    )
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, 
        T_max=scheduler_steps,
        eta_min=1e-7
    )
    
    scaler = torch.amp.GradScaler('cuda') if Config.DEVICE == 'cuda' else None
    step = 0
    print("Start Training...")
    
    for epoch in range(Config.EPOCHS):
        net_g.train()
        net_d.train()
        
        use_gan = epoch >= Config.WARMUP_EPOCHS
        
        for i, x in enumerate(dataloader):
            x = x.to(Config.DEVICE, non_blocking=True) # non_blocking for faster transfer
            step += 1
            N, _, H, W = x.size()
            num_pixels = N * H * W
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()
            optimizer_aux.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                out_net = net_g(x)
                x_hat = out_net['x_hat']
                likelihoods = out_net['likelihoods']
                
                bpp_loss = sum(torch.log(l).sum() / (-0.6931 * num_pixels) for l in likelihoods.values())
                mse_loss = mse_criterion(x_hat, x)
                
                p_loss = torch.tensor(0.0, device=Config.DEVICE)
                if use_gan:
                    p_loss = lpips_criterion((x_hat * 2 - 1), (x * 2 - 1)).mean()
                
                g_loss = torch.tensor(0.0, device=Config.DEVICE)
                fm_loss = torch.tensor(0.0, device=Config.DEVICE)
                
                if use_gan:
                    pred_fake_list, feats_fake_list = net_d(x_hat)
                    for pred_fake in pred_fake_list:
                        g_loss += torch.mean((pred_fake - 1) ** 2)
                    
                    with torch.no_grad():
                        _, feats_real_list = net_d(x)
                    
                    feat_weights = 4.0 / (len(feats_fake_list) * len(feats_fake_list[0]))
                    for list_fake, list_real in zip(feats_fake_list, feats_real_list):
                        for feat_f, feat_r in zip(list_fake, list_real):
                            fm_loss += l1_criterion(feat_f, feat_r.detach()) * feat_weights

                total_loss = (Config.LAMBDA_RATE * bpp_loss + 
                              Config.LAMBDA_MSE * mse_loss + 
                              Config.LAMBDA_LPIPS * p_loss + 
                              Config.LAMBDA_GAN * g_loss + 
                              Config.LAMBDA_FM * fm_loss)
            
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
                scaler.step(optimizer_g)
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
                optimizer_g.step()
            
            # Aux update
            aux_loss = 0
            for m in net_g.modules():
                if isinstance(m, EntropyBottleneck): aux_loss += m.loss()
            aux_loss.backward()
            optimizer_aux.step()
            optimizer_aux.zero_grad()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_loss_val = 0.0
            if use_gan:
                optimizer_d.zero_grad()
                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    pred_real_list, _ = net_d(x)
                    pred_fake_list, _ = net_d(x_hat.detach())
                    
                    loss_d = 0.0
                    for pred_real, pred_fake in zip(pred_real_list, pred_fake_list):
                        loss_d += 0.5 * (torch.mean((pred_real - 1) ** 2) + torch.mean(pred_fake ** 2))
                
                if scaler:
                    scaler.scale(loss_d).backward()
                    scaler.step(optimizer_d)
                else:
                    loss_d.backward()
                    optimizer_d.step()
                d_loss_val = loss_d.item()
            
            if scaler: scaler.update()
            
            if step % 50 == 0:
                current_psnr = calc_psnr(x, x_hat)
                lr_curr = optimizer_g.param_groups[0]['lr']
                status = "GAN_Active" if use_gan else "Warmup"
                print(f"[{status}] Ep {epoch} | LR: {lr_curr:.2e} | "
                      f"Tot: {total_loss.item():.2f} | PSNR: {current_psnr:.2f} | "
                      f"G_Loss: {g_loss.item():.3f} | D_Loss: {d_loss_val:.3f}")

        if use_gan:
            scheduler_g.step()
            scheduler_d.step()

        # Save Checkpoint
        save_path = f"checkpoints/hific_icnr_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'g_state_dict': net_g.state_dict(),
            'd_state_dict': net_d.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'scheduler_g': scheduler_g.state_dict()
        }, save_path)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted.")