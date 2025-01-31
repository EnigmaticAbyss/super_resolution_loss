import torch
import torch.nn as nn
import torch.fft as fft

class FourierDifferencePerceptualLoss(nn.Module):
    def __init__(self, perceptual_loss_fn=None):
        """
        Fourier Difference-based Perceptual Loss.
        Doing perceptual loss  on difference of phase and magnitute of super resolution and ground truth image.
        Args:
        - perceptual_loss_fn (callable): Function to compute perceptual loss (e.g., L1Loss, VGG-based).
        """
        super(FourierDifferencePerceptualLoss, self).__init__()
        
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
        Compute the Fourier Difference-based Perceptual Loss between SR and HR images.
        
        Args:
        - sr_image (torch.Tensor): Super-resolution image (batch, channels, height, width).
        - hr_image (torch.Tensor): High-resolution image (batch, channels, height, width).
        
        Returns:
        - loss (torch.Tensor): Perceptual loss based on Fourier differences.
        """
        
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        sr_image = sr_image.to(device)
        hr_image = hr_image.to(device)
        # Compute magnitude and phase spectra for SR and HR images
        sr_mag, sr_phase = self.compute_magnitude_phase(sr_image)
        hr_mag, hr_phase = self.compute_magnitude_phase(hr_image)
        
        # Compute differences
        mag_diff = sr_mag - hr_mag
        phase_diff = sr_phase - hr_phase
        
        # Compute perceptual loss on the differences
        loss = self.perceptual_loss_fn(mag_diff, phase_diff)
        
        return loss
