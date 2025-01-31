import torch
import torch.nn as nn
import torch.fft as fft


class FourierPerceptualLoss(nn.Module):
    def __init__(self, perceptual_loss_fn=None):
        """
        Fourier-based Perceptual Loss.
        First do fourier and then then perceptual loss for it
        Args:
        - perceptual_loss_fn (callable): Function to compute perceptual loss (e.g., L1Loss, VGG-based).
        """
        super(FourierPerceptualLoss, self).__init__()
        
        # Use default L1 loss if no perceptual loss function is provided
        self.perceptual_loss_fn = perceptual_loss_fn if perceptual_loss_fn else nn.L1Loss()

    def compute_magnitude_phase(self, image):
        """
        Compute magnitude and phase spectra from the Fourier transform of an image.
        
        Args:
        - image (torch.Tensor): Input image (batch, channels, height, width).
        
        Returns:
        - magnitude (torch.Tensor): Magnitude spectrum.
        - phase (torch.Tensor): Phase spectrum.
        """
        # Compute the 2D FFT
        fft_image = fft.fft2(image)
        fft_image_shifted = fft.fftshift(fft_image, dim=(-2, -1))  # Shift zero frequency to center
        
        # Magnitude and phase
        magnitude = torch.abs(fft_image_shifted)
        phase = torch.angle(fft_image_shifted)
        
        return magnitude, phase

    def forward(self, sr_image, hr_image):
        """
        Compute the Fourier-based perceptual loss between SR and HR images.
        
        Args:
        - sr_image (torch.Tensor): Super-resolution image (batch, channels, height, width).
        - hr_image (torch.Tensor): High-resolution image (batch, channels, height, width).
        
        Returns:
        - loss (torch.Tensor): Combined perceptual loss for magnitude and phase spectra.
        """
    # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        sr_image = sr_image.to(device)
        hr_image = hr_image.to(device)
        
        # Compute magnitude and phase spectra for SR and HR images
        sr_mag, sr_phase = self.compute_magnitude_phase(sr_image)
        hr_mag, hr_phase = self.compute_magnitude_phase(hr_image)
        
        # Compute perceptual loss for magnitude and phase spectra
        mag_loss = self.perceptual_loss_fn(sr_mag, hr_mag)
        phase_loss = self.perceptual_loss_fn(sr_phase, hr_phase)
        
        # Combine losses (can be weighted if desired)
        total_loss = mag_loss + phase_loss
        
        return total_loss
