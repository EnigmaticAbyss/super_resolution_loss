
# import torch
# import torch.nn as nn
# import torch.fft as fft
# import torchvision.models as models

# from torch.nn.functional import unfold, fold
# from hiera import Hiera





# class HieraPerceptualLoss(nn.Module):
#     def __init__(self, layers=[ 'blocks.2.norm2'], pretrained=True, device='cuda', patch_size=64, stride=32):
#         super(HieraPerceptualLoss, self).__init__()
#         self.device = device
#         self.patch_size = patch_size
#         self.stride = stride

#         # Load pre-trained Hiera model
#         self.hiera = Hiera.from_pretrained('facebook/hiera_base_224.mae_in1k_ft_in1k').to(device).eval()
#         for param in self.hiera.parameters():
#             param.requires_grad = False

#         # Specify layers for feature extraction
#         self.layers = layers
#         self.selected_layers = self.get_hiera_layers(self.hiera, layers)

#         # Define MSE loss
#         self.mse_loss = nn.MSELoss()

#     def get_hiera_layers(self, model, layers):
#         """Extract specific layers from the Hiera model."""
#         selected_layers = nn.ModuleList()
#         for name, layer in model.named_modules():

#             if name in layers:
   
#                 selected_layers.append(layer)
#                 break
#         return selected_layers


#     def compute_features(self, image):
#         """Extract features from the selected Hiera layers."""
#         features = []
#         x = image
#         for layer in self.selected_layers:
#             x = layer(x)
#             features.append(x)
#         return features



#     def compute_magnitude_phase(self, features):
#         """Compute magnitude and phase spectra from the Fourier transform of features."""
#         magnitudes, phases = [], []
#         for feature in features:


#             # # Remove the DC component (mean) for numerical stability
#             # feature = feature - feature.mean(dim=(-2, -1), keepdim=True)



#             fft_feature = fft.fft2(feature, dim=(-2, -1))
#             fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
#             magnitude = torch.abs(fft_feature_shifted)
#             phase = torch.angle(fft_feature_shifted)
#             # phase = torch.angle(fft_feature_shifted + 1e-8)  # Add epsilon to avoid instability

#             magnitudes.append(magnitude)
#             phases.append(phase)
#         return magnitudes, phases
#     def compute_patch_loss(self, sr_patch, hr_patch, alpha=0.5):
#         """Compute loss for a pair of batch."""

#         sr_features = self.compute_features(sr_patch)
#         hr_features = self.compute_features(hr_patch)

#         # Compute magnitude and phase spectra for features
#         sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
#         hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)

#         # Compute magnitude and phase differences
#         mag_loss, phase_loss = 0, 0
#         for sr_m, hr_m, sr_p, hr_p in zip(sr_mag, hr_mag, sr_phase, hr_phase):




#             mag_loss += torch.mean(torch.abs(sr_m - hr_m))  # Magnitude difference
#             phase_loss += torch.mean(torch.abs(sr_p - hr_p))  # Phase difference

#         # Compute perceptual loss
#         perceptual_loss = alpha * mag_loss + (1 - alpha) * phase_loss
#         # print losses
        

#         # Compute MSE loss
#         mse_loss = self.mse_loss(sr_patch, hr_patch)

#         # Combine losses
#         total_loss = perceptual_loss + mse_loss
#         return total_loss


#     def forward(self, sr_image, hr_image, alpha=0.5, beta=0.5):
#         """Compute combined loss (Fourier Feature Perceptual + MSE) between SR and HR images."""
#         # Ensure inputs are on the same device
#         sr_image = sr_image.to(self.device)
#         hr_image = hr_image.to(self.device)

#         # Compute loss for each patch and aggregate
#         total_loss = 0
#         # for sr_patch, hr_patch in zip(sr_patches, hr_patches):
#         for sr_patch, hr_patch in zip(sr_image, hr_image):
#             patch_loss = self.compute_patch_loss(sr_patch, hr_patch, alpha)
#             total_loss += patch_loss

#         # Average over all patches
#         # total_loss /= sr_patches.size(1)  # Normalize by number of patches
#         return total_loss



# import torch
# import torch.nn as nn
# import torch.fft as fft
# from hiera import hiera_base_224  # Load pre-trained Hiera model


# class HieraPerceptualLoss(nn.Module):
#     def __init__(self, layers=[2], pretrained=True, device='cuda', alpha=0.5):
#         """
#         Computes perceptual loss using Hiera model's intermediate features and Fourier transform.
        
#         Args:
#         - layers (list): Indices of HieraBlocks from which to extract features.
#         - pretrained (bool): Whether to use a pretrained Hiera model.
#         - device (str): Device to run the computations (default: 'cuda').
#         - alpha (float): Weighting factor for magnitude vs. phase loss.
#         """
#         super().__init__()
#         self.device = device
#         self.alpha = alpha
        
#         # Load pre-trained Hiera model and freeze parameters
#         self.hiera = hiera_base_224(pretrained=True).to(device).eval()
#         for param in self.hiera.parameters():
#             param.requires_grad = False

#         # Select specific Hiera blocks for feature extraction
#         self.selected_layers = layers
#         self.mse_loss = nn.MSELoss()

#     def compute_features(self, x):
#         """ Extract features from the selected Hiera blocks. """
#         features = []
#         print("image shape")
#         print(x.shape)
#         x = self.hiera.patch_embed(x)  # Patch embedding
#         print("Embedding")
#         print(x.shape)
#         print("Pos embeded")
#         x = x + self.hiera.get_pos_embed()
#         print(x.shape)
#         print("UNROLL")
#         x = self.hiera.unroll(x)
#         print(x.shape)
#         # Extract intermediate features
#         for i, block in enumerate(self.hiera.blocks):
#             print("block")
#             x = block(x)
#             print(x.shape)
#             if i in self.selected_layers:
#                 features.append(x)
        
#         return features

#     def compute_magnitude_phase(self, features):
#         """ Compute magnitude and phase spectra using Fourier transform. """
#         magnitudes, phases = [], []
#         for feature in features:
#             fft_feature = fft.fft2(feature, dim=(-2, -1))  # 2D FFT
#             fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
#             magnitudes.append(torch.abs(fft_feature_shifted))
#             phases.append(torch.angle(fft_feature_shifted))
#         return magnitudes, phases

#     def compute_patch_loss(self, sr_patch, hr_patch):
#         """ Compute perceptual loss for a pair of image patches. """
#         print("start sr features- calculation")
#         sr_features = self.compute_features(sr_patch)
#         hr_features = self.compute_features(hr_patch)
        
#         sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
#         hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)
        
#         mag_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_mag, hr_mag))
#         phase_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_phase, hr_phase))
#         perceptual_loss = self.alpha * mag_loss + (1 - self.alpha) * phase_loss

#         return perceptual_loss + self.mse_loss(sr_patch, hr_patch)  # Combine with MSE loss

#     def forward(self, sr_image, hr_image):
#         """ Compute combined perceptual loss between SR and HR images. """
#         sr_image, hr_image = sr_image.to(self.device), hr_image.to(self.device)
#         total_loss = sum(self.compute_patch_loss(sr, hr) for sr, hr in zip(sr_image, hr_image))
#         return total_loss


# import torch
# import torch.nn as nn
# import torch.fft as fft
# import torch.nn.functional as F
# from hiera import hiera_base_224  # Load pre-trained Hiera model

# class HieraPerceptualLoss(nn.Module):
#     def __init__(self, layers=[1], device='cuda', alpha=0.5):
#         """
#         Computes perceptual loss using intermediate features from Hiera.
        
#         Args:
#         - layers (list): Indices of Hiera stages to extract features from.
#         - device (str): Computation device (default: 'cuda').
#         - alpha (float): Weighting factor for magnitude vs. phase loss.
#         """
#         super().__init__()
#         self.device = device
#         self.alpha = alpha

#         # Load Pretrained Hiera Model
#         self.hiera = hiera_base_224(pretrained=True).to(device).eval()
#         for param in self.hiera.parameters():
#             param.requires_grad = False  # Freeze model weights

#         self.selected_layers = layers
#         self.mse_loss = nn.MSELoss()

#     def compute_features(self, x):
#         """ Extract features from Hiera using return_intermediates. """
#         print("shape")
#         print(x.shape)
#         x, intermediates = self.hiera(x, return_intermediates=True)
#         print("Intermediate")
#         print(intermediates)

#         print(intermediates.shape)
#         selected_features = [intermediates[i] for i in self.selected_layers]  # Pick only needed layers
#                 # Debugging: Print extracted feature shapes
#         for i, feat in enumerate(selected_features):
#             print(f"Feature {i} shape: {feat.shape}")
#         print("selected")
#         print(selected_features.shape)
#         return selected_features

#     def resize_features(self, features, target_shape):
#         """ Resize feature maps to a consistent spatial size. """
#         return [F.interpolate(f, size=target_shape[-2:], mode='bilinear', align_corners=False) for f in features]

#     def compute_magnitude_phase(self, features):
#         """ Compute magnitude and phase using Fourier Transform. """
#         magnitudes, phases = [], []
#         for feature in features:
#             fft_feature = fft.fft2(feature, dim=(-2, -1))
#             fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
#             magnitudes.append(torch.abs(fft_feature_shifted))
#             phases.append(torch.angle(fft_feature_shifted))
#         return magnitudes, phases

#     def compute_patch_loss(self, sr_patch, hr_patch):
#         """ Compute perceptual loss for a pair of patches. """
#         print(sr_patch.shape)
#         print(hr_patch.shape)
#         sr_features = self.compute_features(sr_patch)
#         hr_features = self.compute_features(hr_patch)


#         print("sr shape")
#         print(sr_features.shape)
#         print("hr shape")
#         print(hr_features.shape)
#         # Resize SR features to match HR feature shape
#         target_shape = hr_features[0].shape  # Match to HR feature size
#         sr_features = self.resize_features(sr_features, target_shape)

#         # Compute Magnitude and Phase loss
#         sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
#         hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)

#         mag_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_mag, hr_mag))
#         phase_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_phase, hr_phase))
#         perceptual_loss = self.alpha * mag_loss + (1 - self.alpha) * phase_loss

#         # MSE Loss for Pixel Matching
#         mse_loss = self.mse_loss(sr_patch, hr_patch)

#         return perceptual_loss + mse_loss  # Combined loss
    
#     def forward(self, sr_image, hr_image):
#         """ Compute perceptual loss between SR and HR images. """

#         sr_image, hr_image = sr_image.to(self.device), hr_image.to(self.device)
#         total_loss = sum(self.compute_patch_loss(sr, hr) for sr, hr in zip(sr_image, hr_image))
#         return total_loss
    
    
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from hiera import hiera_base_224  # Load pre-trained Hiera model

class HieraPerceptualLoss(nn.Module):
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
            fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
            magnitudes.append(torch.abs(fft_feature_shifted))
            phases.append(torch.angle(fft_feature_shifted))
        return magnitudes, phases

    def compute_patch_loss(self, sr_batch, hr_batch):
        """ Compute perceptual loss for a batch of images. """

        # Compute features for entire batch
        sr_features = self.compute_features(sr_batch)
        hr_features = self.compute_features(hr_batch)

        # Resize SR features to match HR feature shape
        # target_shape = hr_features[0].shape  # Match to HR feature size
        # sr_features = self.resize_features(sr_features, target_shape)

        # Compute Magnitude and Phase loss
        sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
        hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)

        mag_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_mag, hr_mag))
        phase_loss = sum(torch.mean(torch.abs(s - h)) for s, h in zip(sr_phase, hr_phase))
        perceptual_loss = self.alpha * mag_loss + (1 - self.alpha) * phase_loss

        # MSE Loss for Pixel Matching
        mse_loss = self.mse_loss(sr_batch, hr_batch)

        return perceptual_loss + mse_loss  # Combined loss
    
    def forward(self, sr_image, hr_image):
        """ Compute perceptual loss for a batch of images. """

        sr_image, hr_image = sr_image.to(self.device), hr_image.to(self.device)

        # Process entire batch at once instead of looping
        total_loss = self.compute_patch_loss(sr_image, hr_image)

        return total_loss
