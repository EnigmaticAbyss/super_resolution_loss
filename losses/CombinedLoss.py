import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """
    Combine frequency loss and perceptual loss into one loss.

    Args:
        freq_loss (torch.Tensor): Computed frequency loss.
        perc_loss (torch.Tensor): Computed perceptual loss.
        alpha (float): Weight for frequency loss.
        beta (float): Weight for perceptual loss.

    Returns:
        torch.Tensor: Combined loss.
    """
    def __init__(self, freq_loss, perc_loss, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.freq_loss = freq_loss
        self.perc_loss = perc_loss
        self.alpha = alpha
        self.beta = beta
 
    def forward(self, sr_image, hr_image): 
        total_loss = self.alpha * self.freq_loss(sr_image,hr_image) + self.beta * self.perc_loss(sr_image,hr_image)
        return total_loss






