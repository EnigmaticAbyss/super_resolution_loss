import torch
import torch.nn as nn
import torch.fft as fft

class FrequencyLoss(nn.Module):
    def __init__(self, loss_type='l1', weight_magnitude=1.0, weight_phase=1.0, device='cuda'):
        """
        Frequency Loss .

        Args:
        - loss_type (str): Type of loss ('l1' or 'l2') to compare magnitude and phase.
        - weight_magnitude (float): Weight for the magnitude loss component.
        - weight_phase (float): Weight for the phase loss component.
        - device (str): Device to perform computations on ('cuda' or 'cpu').
        """
        super(FrequencyLoss, self).__init__()
        assert loss_type in ['l1', 'l2'], "loss_type must be 'l1' or 'l2'"
        self.loss_type = loss_type
        self.weight_magnitude = weight_magnitude
        self.weight_phase = weight_phase
        self.device = torch.device(device)

        # Define the base loss function
        self.base_loss = nn.L1Loss() if loss_type == 'l1' else nn.MSELoss()

    def compute_magnitude_phase(self, image):
        """
        Compute magnitude and phase spectra from the Fourier transform of an image.

        Args:
        - image (torch.Tensor): Input image (batch, channels, height, width).

        Returns:
        - magnitude (torch.Tensor): Magnitude spectrum.
        - phase (torch.Tensor): Phase spectrum.
        """
        # Compute 2D FFT
        fft_image = fft.fft2(image)
        fft_image_shifted = fft.fftshift(fft_image, dim=(-2, -1))  # Shift zero frequency to center

        # Magnitude and phase
        magnitude = torch.abs(fft_image_shifted)
        phase = torch.angle(fft_image_shifted)

        return magnitude, phase

    def forward(self, sr_image, hr_image):
        """
        Compute the Frequency Loss between SR and HR images.

        Args:
        - sr_image (torch.Tensor): Super-resolution image (batch, channels, height, width).
        - hr_image (torch.Tensor): High-resolution image (batch, channels, height, width).

        Returns:
        - loss (torch.Tensor): Combined loss for magnitude and phase differences.
        """
        # Ensure the tensors are on the correct device
        sr_image = sr_image.to(self.device)
        hr_image = hr_image.to(self.device)

        # Compute magnitude and phase for SR and HR images
        sr_mag, sr_phase = self.compute_magnitude_phase(sr_image)
        hr_mag, hr_phase = self.compute_magnitude_phase(hr_image)

        # Compute loss for magnitude and phase differences
        mag_loss = self.base_loss(sr_mag, hr_mag)
        phase_loss = self.base_loss(sr_phase, hr_phase)

        # Weighted sum of losses
        total_loss = self.weight_magnitude * mag_loss + self.weight_phase * phase_loss
        return total_loss
