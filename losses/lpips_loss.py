#LPIPS (Learned Perceptual Image Patch Similarity)

import lpips
import torch

class LPIPSLoss:
    def __init__(self, device):
        self.lpips = lpips.LPIPS(net='vgg').to(device)
    
    def __call__(self, sr, hr):
        return self.lpips(sr, hr).mean()
