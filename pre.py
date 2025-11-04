import torch 
import torch.nn.functional as F
import numpy as np
class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)
block = GeneratorBlock(in_dim=256, out_dim=128, kernel=(2,4,4), stride=(1,2,2))
x = torch.randn(1, 256, 1, 4, 4)  # entrée bruitée (1 échantillon, 256 canaux)
y = block(x)
print(y.shape)
