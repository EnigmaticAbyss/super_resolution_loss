
import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.models as models
from torch.nn.functional import unfold, fold


class FourierFeaturePerceptualLoss(nn.Module):
    """First doing perecptual loss on patches and then do some fourier loss on them as well """
    def __init__(self, layers=[ 'relu_4'], use_pretrained=True, device='cuda', patch_size=64, stride=32):
        super(FourierFeaturePerceptualLoss, self).__init__()
        self.device = device
        self.patch_size = patch_size 
        self.stride = stride       
        
        # # Mean and std for ImageNet normalization
        # self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        # self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # Define MSE loss
        self.mse_loss = nn.MSELoss()

        # Load pre-trained VGG model and freeze parameters
        vgg = models.vgg19(pretrained=use_pretrained).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract specified layers for perceptual loss
        self.layers = layers
        self.selected_layers = self.get_vgg_layers(vgg, layers)
        




        
    # def preprocess_for_vgg(self, image):
    #     """
    #     Normalize input image to match VGG's expected input range.

    #     Args:
    #     - image (torch.Tensor): Input image (batch, channels, height, width), values in [0, 1].

    #     Returns:
    #     - normalized_image (torch.Tensor): Normalized image.
    #     """
    #     return (image - self.imagenet_mean) / self.imagenet_std

    def extract_patches(self, image):
        """Divide an image into patches."""
        b, c, h, w = image.size()
        patch_image = unfold(image, kernel_size=self.patch_size, stride=self.stride)
        # Reshape to (batch, channels, patch_size, patch_size, num_patches)
        patch_image = patch_image.permute(0, 2, 1).reshape(b, -1, c, self.patch_size, self.patch_size)
        return patch_image


    def get_vgg_layers(self, model, layers):
        """Extract specific layers from the VGG model."""
        selected_layers = nn.ModuleList()
        layer_mapping = {'relu_1': 0, 'relu_2': 5, 'relu_3': 10, 'relu_4': 19, 'relu_5': 28}
        for name in layers:
            if name in layer_mapping:
                selected_layers.append(model[:layer_mapping[name] + 1])
            else:
                raise ValueError(f"Layer {name} not found in VGG model.")
        return selected_layers


    def normalize_to_match_mean_std(self, sr_image, hr_image):
        """Normalize SR image to have the same mean and std as the HR image."""
        sr_mean = sr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
        sr_std = sr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel
        
        hr_mean = hr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
        hr_std = hr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel

        # Normalize SR image to have same mean and std as HR image
        sr_image_normalized = (sr_image - sr_mean) / (sr_std + 1e-8)  # Normalize SR
        sr_image_normalized = sr_image_normalized * hr_std + hr_mean  # Match HR stats

        return sr_image_normalized





    def compute_features(self, image):
        """Extract features from selected VGG layers."""
        features = []
        x = image
        for layer in self.selected_layers:
            x = layer(x)
            features.append(x)
            x = image

        return features

    def compute_magnitude_phase(self, features):
        """Compute magnitude and phase spectra from the Fourier transform of features."""
        magnitudes, phases = [], []
        for feature in features:


            # # Remove the DC component (mean) for numerical stability
            # feature = feature - feature.mean(dim=(-2, -1), keepdim=True)



            fft_feature = fft.fft2(feature, dim=(-2, -1))
            fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
            magnitude = torch.abs(fft_feature_shifted)
            phase = torch.angle(fft_feature_shifted)
            # phase = torch.angle(fft_feature_shifted + 1e-8)  # Add epsilon to avoid instability

            magnitudes.append(magnitude)
            phases.append(phase)
        return magnitudes, phases
    def compute_patch_loss(self, sr_patch, hr_patch, alpha=0.5):
        """Compute loss for a pair of patches."""
        # # Normalize SR image to match HR image statistics
        # sr_patch = self.normalize_to_match_mean_std(sr_patch, hr_patch)

        # # Preprocess patches for VGG
        # sr_patch_vgg = self.preprocess_for_vgg(sr_patch)
        # hr_patch_vgg = self.preprocess_for_vgg(hr_patch)

        # Extract features from patches
        # sr_features = self.compute_features(sr_patch_vgg)
        # hr_features = self.compute_features(hr_patch_vgg)
        sr_features = self.compute_features(sr_patch)
        hr_features = self.compute_features(hr_patch)

        # Compute magnitude and phase spectra for features
        sr_mag, sr_phase = self.compute_magnitude_phase(sr_features)
        hr_mag, hr_phase = self.compute_magnitude_phase(hr_features)

        # Compute magnitude and phase differences
        mag_loss, phase_loss = 0, 0
        for sr_m, hr_m, sr_p, hr_p in zip(sr_mag, hr_mag, sr_phase, hr_phase):

            ### this is for mag_mean of batch per channel of group **2 ###
            # # Compute mean and std of magnitude and phase differences per channel
            # mag_diff = sr_m - hr_m
            # phase_diff = sr_p - hr_p
 
 
            # mag_mean = torch.mean(mag_diff, dim=(-2, -1))  # Mean per channel
            # mag_std = torch.std(mag_diff, dim=(-2, -1))    # Std per channel

            # phase_mean = torch.mean(phase_diff, dim=(-2, -1))
            # phase_std = torch.std(phase_diff, dim=(-2, -1))

            # # Aggregate loss across channels
            # mag_loss += torch.mean(mag_mean ** 2 + mag_std ** 2)  # MSE of mean and std
            # phase_loss += torch.mean(phase_mean ** 2 + phase_std ** 2)



            mag_loss += torch.mean(torch.abs(sr_m - hr_m))  # Magnitude difference
            phase_loss += torch.mean(torch.abs(sr_p - hr_p))  # Phase difference

        # Compute perceptual loss
        perceptual_loss = alpha * mag_loss + (1 - alpha) * phase_loss
        # print losses
        

        # Compute MSE loss
        mse_loss = self.mse_loss(sr_patch, hr_patch)

        # Combine losses
        total_loss = perceptual_loss + mse_loss
        return total_loss


    def forward(self, sr_image, hr_image, alpha=0.5, beta=0.5):
        """Compute combined loss (Fourier Feature Perceptual + MSE) between SR and HR images."""
        # Ensure inputs are on the same device
        sr_image = sr_image.to(self.device)
        hr_image = hr_image.to(self.device)

        # # Divide images into patches
        # sr_patches = self.extract_patches(sr_image)
        # hr_patches = self.extract_patches(hr_image)

        # Compute loss for each patch and aggregate
        total_loss = 0
        # for sr_patch, hr_patch in zip(sr_patches, hr_patches):
        for sr_patch, hr_patch in zip(sr_image, hr_image):
            patch_loss = self.compute_patch_loss(sr_patch, hr_patch, alpha)
            total_loss += patch_loss

        # Average over all patches
        # total_loss /= sr_patches.size(1)  # Normalize by number of patches
        return total_loss









# import torch
# import torch.nn as nn
# import torch.fft as fft
# import torchvision.models as models
# from torch.nn.functional import unfold, fold

# class FourierFeaturePerceptualLoss(nn.Module):
#     def __init__(self, layers=['relu_3', 'relu_4'], use_pretrained=True, device='cuda', patch_size=64, stride=32):
#         super(FourierFeaturePerceptualLoss, self).__init__()
#         self.device = device
#         self.patch_size = patch_size
#         self.stride = stride

#         # Load pre-trained VGG model and freeze parameters
#         vgg = models.vgg19(pretrained=use_pretrained).features.to(device).eval()
#         for param in vgg.parameters():
#             param.requires_grad = False

#         # Extract specified layers for perceptual loss
#         self.layers = layers
#         self.selected_layers = self.get_vgg_layers(vgg, layers)

#         # Define MSE loss
#         self.mse_loss = nn.MSELoss()

#         # Define ImageNet normalization parameters
#         self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
#         self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

#     def get_vgg_layers(self, model, layers):
#         """Extract specific layers from the VGG model."""
#         selected_layers = nn.ModuleList()
#         layer_mapping = {'relu_1': 0, 'relu_2': 5, 'relu_3': 10, 'relu_4': 19, 'relu_5': 28}
#         for name in layers:
#             if name in layer_mapping:
#                 selected_layers.append(model[:layer_mapping[name] + 1])
#             else:
#                 raise ValueError(f"Layer {name} not found in VGG model.")
#         return selected_layers

#     def normalize_to_match_mean_std(self, sr_image, hr_image):
#         """Normalize SR image to have the same mean and std as the HR image."""
#         sr_mean = sr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
#         sr_std = sr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel
        
#         hr_mean = hr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
#         hr_std = hr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel

#         # Normalize SR image to have same mean and std as HR image
#         sr_image_normalized = (sr_image - sr_mean) / (sr_std + 1e-8)  # Normalize SR
#         sr_image_normalized = sr_image_normalized * hr_std + hr_mean  # Match HR stats

#         return sr_image_normalized

#     def preprocess_for_vgg(self, image):
#         """Normalize image to ImageNet mean and std for VGG."""
#         return (image - self.imagenet_mean) / self.imagenet_std

#     def extract_patches(self, image):
#         """Divide an image into patches."""
#         b, c, h, w = image.size()
#         patch_image = unfold(image, kernel_size=self.patch_size, stride=self.stride)
#         # Reshape to (batch, channels, patch_size, patch_size, num_patches)
#         patch_image = patch_image.permute(0, 2, 1).reshape(b, -1, c, self.patch_size, self.patch_size)
#         return patch_image

#     def compute_features(self, image):
#         """Extract features from selected VGG layers."""
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
#             fft_feature = fft.fft2(feature, dim=(-2, -1))
#             fft_feature_shifted = fft.fftshift(fft_feature, dim=(-2, -1))
#             magnitude = torch.abs(fft_feature_shifted)
#             phase = torch.angle(fft_feature_shifted)
#             magnitudes.append(magnitude)
#             phases.append(phase)
#         return magnitudes, phases

#     def compute_patch_loss(self, sr_patch, hr_patch, alpha=0.5):
#         """Compute loss for a pair of patches."""
#         # Normalize SR image to match HR image statistics
#         sr_patch = self.normalize_to_match_mean_std(sr_patch, hr_patch)

#         # Preprocess patches for VGG
#         sr_patch_vgg = self.preprocess_for_vgg(sr_patch)
#         hr_patch_vgg = self.preprocess_for_vgg(hr_patch)

#         # Extract features from patches
#         sr_features = self.compute_features(sr_patch_vgg)
#         hr_features = self.compute_features(hr_patch_vgg)

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

#         # Divide images into patches
#         sr_patches = self.extract_patches(sr_image)
#         hr_patches = self.extract_patches(hr_image)

#         # Compute loss for each patch and aggregate
#         total_loss = 0
#         for sr_patch, hr_patch in zip(sr_patches, hr_patches):
#             patch_loss = self.compute_patch_loss(sr_patch, hr_patch, alpha)
#             total_loss += patch_loss

#         # Average over all patches
#         total_loss /= sr_patches.size(1)  # Normalize by number of patches
#         return total_loss
