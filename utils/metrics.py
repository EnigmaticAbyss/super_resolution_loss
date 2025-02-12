# import torch
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# def calculate_psnr(sr, hr, max_pixel=1.0):
#     """Calculates PSNR for a batch of images."""
#     # Ensure both sr and hr have shape [B, C, H, W]
#     assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"
    
#     psnr_values = []
#     for i in range(sr.shape[0]):
#         mse = torch.mean((sr[i] - hr[i]) ** 2)
#         if mse == 0:
#             psnr_values.append(float("inf"))
#         else:
#             psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
#             psnr_values.append(psnr.item())
    
#     return psnr_values  # Return list of PSNR values for each image in the batch

# def calculate_ssim(sr, hr):
#     """Calculates SSIM for a batch of images."""
#     # Ensure both sr and hr have shape [B, C, H, W]
#     assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"

#     ssim_values = []
#     for i in range(sr.shape[0]):
#         # Convert each image from [C, H, W] to [H, W, C] and then to numpy
#         sr_np = sr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
#         hr_np = hr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
#         # print("first")
#         # print(sr_np.shape)
#         # print("secod")
#         # print(hr_np.shape)
        
        
#         # Compute SSIM for each image
   
# # Use channel_axis for recent versions of skimage (instead of multichannel)
#         ssim_value = ssim(sr_np, hr_np, channel_axis=-1, data_range=hr_np.max() - hr_np.min())
#         ssim_values.append(ssim_value)
    
#     return ssim_values  # Return list of SSIM values for each image in the batch


import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
from torch_fidelity import calculate_metrics

def calculate_psnr(sr, hr, max_pixel=1.0):
    """Calculates PSNR for a batch of images."""
    assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"
    
    psnr_values = []
    for i in range(sr.shape[0]):
        mse = torch.mean((sr[i] - hr[i]) ** 2)
        if mse == 0:
            psnr_values.append(float("inf"))
        else:
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            psnr_values.append(psnr.item())
    
    return psnr_values

def calculate_ssim(sr, hr):
    """Calculates SSIM for a batch of images."""
    assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"

    ssim_values = []
    for i in range(sr.shape[0]):
        sr_np = sr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        hr_np = hr[i].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        
        # Compute SSIM for each image
        ssim_value = ssim(sr_np, hr_np, channel_axis=-1, data_range=hr_np.max() - hr_np.min())
        ssim_values.append(ssim_value)
    
    return ssim_values

# def calculate_fid_score(sr, hr, device="cuda"):
#     """Calculates FID for a batch of images using torch-fidelity."""
# import torch
# from torch_fidelity import calculate_metrics

def calculate_fid_score(sr, hr, device="cuda"):
    """
    Calculates FID for a batch of images using torch-fidelity.
    
    Args:
        sr (torch.Tensor): Super-resolved images tensor of shape [B, C, H, W].
        hr (torch.Tensor): High-resolution images tensor of shape [B, C, H, W].
        device (str): Device to perform computations on ("cuda" or "cpu").
    
    Returns:
        float: Computed FID score.
    """
    assert sr.shape == hr.shape, "Super-resolved and high-res images must have the same shape"
    
    # Move tensors to the specified device
    sr = sr.to(device)
    hr = hr.to(device)
    
    # Normalize images to [0, 1] range if they are not already
    if sr.max() > 1.0 or hr.max() > 1.0:
        sr = sr / 255.0
        hr = hr / 255.0

    # Create a dictionary of input batches
    inputs = {
        'samples1': hr,
        'samples2': sr,
    }

    # Calculate FID using torch-fidelity
    metrics = calculate_metrics(
        input1=inputs['samples1'],
        input2=inputs['samples2'],
        fid = True,
        device=device
    )
 
    # Extract FID score
    fid_score = metrics
    
    return fid_score


def calculate_lpips_score(sr, hr, device="cuda"):
    """Calculates LPIPS for a batch of images."""
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex')  # 'alex' is a common option for LPIPS

    lpips_values = []
    for i in range(sr.shape[0]):
        sr_np = sr[i].cpu().detach().unsqueeze(0)  # [1, C, H, W]
        hr_np = hr[i].cpu().detach().unsqueeze(0)  # [1, C, H, W]
        
        # Compute LPIPS
        sr = sr_np.to(device)
        hr = hr_np.to(device)
        lpips_value = lpips_model(sr, hr)
        lpips_values.append(lpips_value.item())

    return lpips_values
