import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(sr, hr, max_pixel=1.0):
    """Calculates PSNR for a batch of images."""
    # Ensure both sr and hr have shape [B, C, H, W]
    assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"
    
    psnr_values = []
    for i in range(sr.shape[0]):
        mse = torch.mean((sr[i] - hr[i]) ** 2)
        if mse == 0:
            psnr_values.append(float("inf"))
        else:
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            psnr_values.append(psnr.item())
    
    return psnr_values  # Return list of PSNR values for each image in the batch

def calculate_ssim(sr, hr):
    """Calculates SSIM for a batch of images."""
    # Ensure both sr and hr have shape [B, C, H, W]
    assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"

    ssim_values = []
    for i in range(sr.shape[0]):
        # Convert each image from [C, H, W] to [H, W, C] and then to numpy
        sr_np = sr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        hr_np = hr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        # print("first")
        # print(sr_np.shape)
        # print("secod")
        # print(hr_np.shape)
        
        
        # Compute SSIM for each image
   
# Use channel_axis for recent versions of skimage (instead of multichannel)
        ssim_value = ssim(sr_np, hr_np, channel_axis=-1, data_range=hr_np.max() - hr_np.min())
        ssim_values.append(ssim_value)
    
    return ssim_values  # Return list of SSIM values for each image in the batch
