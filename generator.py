import torch
latent_dim=128
class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.trc = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.trc(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)
class Generator(torch.nn.Module):   
    def __init__(self):
        super().__init__()
        self.trc0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.trc1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.trc2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.trc3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.trc4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.trc5 = torch.nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x