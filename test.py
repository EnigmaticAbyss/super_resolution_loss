# # from hiera import Hiera
# # import torch.nn as nn

# # hiera = Hiera.from_pretrained('facebook/hiera_base_224.mae_in1k_ft_in1k')
# # # def get_hiera_layers( model, target_layer):
# # #     """Extract all layers up to the specified target layer."""
# # #     selected_layers = nn.ModuleList()
# # #     for  name,layer in model.named_modules():
# # #         selected_layers.append(layer)
# # #         if layer == target_layer:
# # #             break  # Stop after the target layer
# # #     return selected_layers

# # def get_hiera_layers( model, layers):
# #     """Extract specific layers from the Hiera model."""
# #     selected_layers = nn.ModuleList()
# #     for name, layer in model.named_modules():
  
# #         if name in layers:

# #             selected_layers.append(layer)
# #             break
# #     return selected_layers

# # print(get_hiera_layers(hiera, layers=['blocks.5']))
# # print("Full model")
# # print(hiera)

# import torch
# from hiera import hiera_base_224

# # Load the pretrained Hiera model
# model = hiera_base_224(pretrained=True).eval()

# # Create a batch of 8 RGB images (each 224x224)
# batch_images = torch.randn(16, 3, 224, 224)  # Batch size = 8


# # print("shape")
# # print(batch_images.shape)
# x, intermediates = model(batch_images, return_intermediates=True)
# print("Intermediate")
# print(model.stage_ends)
# # print(intermediates)
# print(len(intermediates))
# # print(intermediates[2].shape)
# # print(x.shape)

# # # Print output shape
# # print("Output Shape:", batch_images.shape)  # Expected: (8, num_classes)



# # from pathlib import Path



# # def handle_empty_config():
# #     """
# #     Handle the case when no configuration files are provided.
# #     """
# #     main_dir = Path(__file__).resolve().parent
# #     saved_model_dir = main_dir / "saved_models"

# #     saved_models = [f for f in saved_model_dir.iterdir() if f.is_file()]

# #     for saved_model in saved_models:
# #         print(f"Processing model: {saved_model}")
# #         # Your evaluation logic here
        
# # handle_empty_config()
import sys
import torch

print("=== PYTHON DEBUG INFO ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("Torch built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("==========================")
