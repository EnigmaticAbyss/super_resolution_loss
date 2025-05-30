import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils  # ADD THIS for grids

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu_3'], device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        
        # Load VGG19 model and freeze its parameters
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract specified layers for perceptual loss
        self.layers = layers
        self.selected_layers = self.get_vgg_layers(vgg, layers)

    def get_vgg_layers(self, model, layers):
        """Extracts specific layers from the VGG model."""
        selected_layers = nn.ModuleList()
        layer_mapping = {
            'relu_1': 0, 'relu_2': 5, 'relu_3': 10, 'relu_4': 19, 'relu_5': 28
        }
        
        for name in layers:
            if name in layer_mapping:
                selected_layers.append(model[:layer_mapping[name] + 1])
            else:
                raise ValueError(f"Layer {name} not found in VGG model.")
        
        return selected_layers

    def forward(self, sr, hr):
        """Computes the perceptual loss between super-resolved and high-res images."""
        sr_features = sr
        hr_features = hr
        loss = 0.0
        
        for layer in self.selected_layers:
            # Pass both super-resolved and high-res images through the VGG layers
            sr_features = layer(sr_features)
            hr_features = layer(hr_features)
            
            # Compute the L1 loss between the feature maps
            loss += torch.nn.functional.l1_loss(sr_features, hr_features)
            # print("Perceptual loss")
            # print(loss.shape)        
        return loss
    def extract_features(self, img):
        """Extract intermediate features from VGG for visualization."""
        features = []
        x = img
        for layer in self.selected_layers:
            x = layer(x)
            features.append(x)
        return features

    def create_feature_grid(self, features, nrow=8):
        """Create a grid of feature maps for TensorBoard."""
        grids = []
        for feat in features:
            # Only take the first sample in batch
            feat = feat[0]  # (C, H, W)
            # Normalize feature maps for visualization
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-5)
            # Make a grid
            grid = vutils.make_grid(feat.unsqueeze(1), nrow=nrow, normalize=False, padding=1)
            grids.append(grid)
        return grids