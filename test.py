# from hiera import Hiera
# import torch.nn as nn

# hiera = Hiera.from_pretrained('facebook/hiera_base_224.mae_in1k_ft_in1k')
# # def get_hiera_layers( model, target_layer):
# #     """Extract all layers up to the specified target layer."""
# #     selected_layers = nn.ModuleList()
# #     for  name,layer in model.named_modules():
# #         selected_layers.append(layer)
# #         if layer == target_layer:
# #             break  # Stop after the target layer
# #     return selected_layers

# def get_hiera_layers( model, layers):
#     """Extract specific layers from the Hiera model."""
#     selected_layers = nn.ModuleList()
#     for name, layer in model.named_modules():
  
#         if name in layers:

#             selected_layers.append(layer)
#             break
#     return selected_layers

# print(get_hiera_layers(hiera, layers=['blocks.5']))
# print("Full model")
# print(hiera)

import torch
from hiera import hiera_base_224

# Load the pretrained Hiera model
model = hiera_base_224(pretrained=True).eval()

# Create a batch of 8 RGB images (each 224x224)
batch_images = torch.randn(16, 3, 224, 224)  # Batch size = 8


print("shape")
print(batch_images.shape)
x, intermediates = model(batch_images, return_intermediates=True)
print("Intermediate")
print(len(intermediates))
print(intermediates[2].shape)
print(x.shape)

# Print output shape
print("Output Shape:", batch_images.shape)  # Expected: (8, num_classes)
