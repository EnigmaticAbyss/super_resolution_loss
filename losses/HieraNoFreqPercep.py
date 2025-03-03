
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from hiera import hiera_base_224  # Load pre-trained Hiera model

class HieraNoFreqPercep(nn.Module):
    def __init__(self, layers=[1], device='cuda', alpha=0.5):
        """
        Computes perceptual loss using intermediate features from Hiera.
        
        Args:
        - layers (list): Indices of Hiera stages to extract features from.
        - device (str): Computation device (default: 'cuda').
        - alpha (float): Weighting factor for magnitude vs. phase loss.
        """
        super().__init__()
        self.device = device
        self.alpha = alpha

        # Load Pretrained Hiera Model
        self.hiera = hiera_base_224(pretrained=True).to(device).eval()
        for param in self.hiera.parameters():
            param.requires_grad = False  # Freeze model weights


        # Ensure selected layers are integers
        self.selected_layers = [int(layer) for layer in layers]  # Convert to integers if needed
        self.mse_loss = nn.MSELoss()


    def compute_features(self, x):
        """ Extract features from Hiera using return_intermediates for batch processing. """
        x, intermediates = self.hiera(x, return_intermediates=True)

        # Extract only the selected layers
        # selected_features = [intermediates[i] for i in self.selected_layers]
        # # Extract selected features based on layer indices (not strings)
        selected_features = [intermediates[i] for i in self.selected_layers if i < len(intermediates)]

        return selected_features  # Each feature has shape (B, C, H, W)

    def resize_features(self, features, target_shape):
        """ Resize feature maps to match target spatial dimensions. """
        return [F.interpolate(f, size=target_shape[-2:], mode='bilinear', align_corners=False) for f in features]

    def compute_magnitude_phase(self, features):
        """ Compute magnitude and phase using Fourier Transform. """
        magnitudes, phases = [], []
        for feature in features:
            fft_feature = fft.fft2(feature, dim=(-2, -1))
            # fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
            magnitudes.append(torch.abs(fft_feature))
            phases.append(torch.angle(fft_feature))
        return magnitudes, phases

    def compute_patch_loss(self, sr_batch, hr_batch):
        """ Compute perceptual loss for a batch of images. """

        # Compute features for entire batch
        sr_features = self.compute_features(sr_batch)
        hr_features = self.compute_features(hr_batch)
        # perceptual_loss = torch.nn.functional.l1_loss(sr_features, hr_features)
        perceptual_loss = sum(torch.nn.functional.l1_loss(s, h) for s, h in zip(sr_features, hr_features)) / len(sr_features)
        
        # Resize SR features to match HR feature shape
        # target_shape = hr_features[0].shape  # Match to HR feature size
        # sr_features = self.resize_features(sr_features, target_shape)

        # # Compute Magnitude and Phase loss
        # sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
        # hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)

        # mag_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_mag, hr_mag))
        # phase_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_phase, hr_phase))
        # perceptual_loss = self.alpha * mag_loss + (1 - self.alpha) * phase_loss

        # MSE Loss for Pixel Matching
        mse_loss = self.mse_loss(sr_batch, hr_batch)
        print(f"MSE Loss: {mse_loss.item()}, Perceptual Loss: {perceptual_loss.item()}")
        lambda_mse = 1.0  
        lambda_perceptual = 0.1  

        # Compute total loss
        total_loss = lambda_mse * mse_loss + lambda_perceptual * perceptual_loss
        
        return total_loss
    def forward(self, sr_image, hr_image):
        """ Compute perceptual loss for a batch of images. """

        sr_image, hr_image = sr_image.to(self.device), hr_image.to(self.device)

        # Process entire batch at once instead of looping
        total_loss = self.compute_patch_loss(sr_image, hr_image)

        return total_loss

