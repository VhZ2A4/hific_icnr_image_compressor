import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from compressai.zoo import mbt2018_mean

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

class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, upscale_kernel=3):
        super(PixelShufflePack, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 
                              kernel_size=upscale_kernel, padding=upscale_kernel//2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.icnr_init(self.conv.weight, scale_factor=scale_factor)
        
    def icnr_init(self, tensor, scale_factor=2, init=nn.init.kaiming_normal_):
        out_channels, in_channels, height, width = tensor.size()
        transpose_in_channels = out_channels // (scale_factor ** 2)
        kernel_shape = (transpose_in_channels, in_channels, height, width)
        kernel = torch.zeros(kernel_shape, dtype=tensor.dtype, device=tensor.device)
        kernel = init(kernel)
        kernel = kernel.contiguous().view(transpose_in_channels, in_channels, -1)
        kernel = kernel.repeat(scale_factor ** 2, 1, 1)
        kernel = kernel.contiguous().view(out_channels, in_channels, height, width)
        with torch.no_grad():
            tensor.copy_(kernel)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class AdvancedRefinementModule(nn.Module):
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        self.conv_in = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_ch, nf, 3, 1, 0), nn.LeakyReLU(0.2, True))
        self.down = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf*2, 3, 2, 0), nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(RRDB(nf*2), RRDB(nf*2), RRDB(nf*2))
        self.up = nn.Sequential(PixelShufflePack(nf*2, nf, scale_factor=2), nn.LeakyReLU(0.2, True))
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3, 1, 0), nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1), nn.Conv2d(nf, in_ch, 3, 1, 0)
        )
        self.skip_conv = nn.Conv2d(nf, nf, 1, 1, 0)

    def forward(self, x):
        f1 = self.conv_in(x)
        f2 = self.down(f1)
        f3 = self.body(f2)
        f4 = self.up(f3) 
        res = self.conv_out(f4 + self.skip_conv(f1))
        return x + res

class BalancedHiFiCGenerator(nn.Module):
    def __init__(self, quality=3):
        super().__init__()
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
        self.layers.append(nn.Sequential(spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)))
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult; nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)), nn.LeakyReLU(0.2, True)))
        nf_mult_prev = nf_mult; nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)), nn.LeakyReLU(0.2, True)))
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
        result = []; feats = []
        for i, d in enumerate(self.discriminators):
            if i > 0: x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            out, feat = d(x)
            result.append(out)
            feats.append(feat)
        return result, feats