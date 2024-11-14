import torch
import torch.nn as nn

class GradientLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        """
        Initializes GradientLoss with a specific type of loss function.
        Args:
            loss_type (str): Type of loss function ('l1' for L1 Loss, 'l2' for L2 Loss).
        """
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, sr, hr):
        """Computes the gradient loss between super-resolved and high-res images."""
        # Compute gradients of SR and HR images
        grad_sr_x, grad_sr_y = self.compute_gradients(sr)
        grad_hr_x, grad_hr_y = self.compute_gradients(hr)

        # Calculate the gradient loss (L1 or L2) for both x and y gradients
        if self.loss_type == 'l1':
            loss = nn.functional.l1_loss(grad_sr_x, grad_hr_x) + nn.functional.l1_loss(grad_sr_y, grad_hr_y)
        elif self.loss_type == 'l2':
            loss = nn.functional.mse_loss(grad_sr_x, grad_hr_x) + nn.functional.mse_loss(grad_sr_y, grad_hr_y)
        else:
            raise ValueError("Unsupported loss type. Use 'l1' or 'l2'.")

        return loss

    @staticmethod
    def compute_gradients(img):
        """Calculates gradients of an image in x and y directions using finite differences."""
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]  # Gradient along x-axis
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]  # Gradient along y-axis

        # Pad to maintain the original image size
        grad_x = nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = nn.functional.pad(grad_y, (0, 0, 0, 1))

        return grad_x, grad_y
