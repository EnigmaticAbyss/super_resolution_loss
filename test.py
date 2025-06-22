# # # # # import torch
# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np
# # # # # from hiera import hiera_base_224

# # # # # # Load the pretrained model
# # # # # model = hiera_base_224(pretrained=True).eval()

# # # # # # Create a batch of dummy images
# # # # # batch_images = torch.randn(16, 3, 224, 224)

# # # # # # Forward pass
# # # # # x, intermediates = model(batch_images, return_intermediates=True)

# # # # # # Choose a sample index (e.g., the first image in the batch)
# # # # # sample_idx = 0

# # # # # def visualize_feature_map(feature_tensor, name="", max_channels=1):
# # # # #     """Visualize up to `max_channels` channels from a feature map tensor."""
# # # # #     if feature_tensor.dim() < 3:
# # # # #         print(f"Skipping {name}: not enough dimensions (dim={feature_tensor.dim()})")
# # # # #         return

# # # # #     # If shape is [B, H, W], add channel dim to make it [B, 1, H, W]
# # # # #     if feature_tensor.dim() == 3:
# # # # #         feature_tensor = feature_tensor.unsqueeze(1)
# # # # #     elif feature_tensor.dim() != 4:
# # # # #         print(f"Skipping {name}: unsupported tensor shape {feature_tensor.shape}")
# # # # #         return

# # # # #     # Select the sample
# # # # #     feature_maps = feature_tensor[sample_idx]  # shape: [C, H, W]
# # # # #     num_channels = feature_maps.shape[0]

# # # # #     # Adjust number of channels to visualize
# # # # #     display_channels = min(max_channels, num_channels)

# # # # #     fig, axes = plt.subplots(1, display_channels, figsize=(5 * display_channels, 5))
# # # # #     if display_channels == 1:
# # # # #         axes = [axes]  # Make iterable

# # # # #     fig.suptitle(name, fontsize=14)

# # # # #     for i in range(display_channels):
# # # # #         fmap = feature_maps[i].detach().cpu().numpy()
# # # # #         # Normalize for visualization
# # # # #         fmap -= fmap.min()
# # # # #         fmap /= fmap.max() + 1e-5
# # # # #         axes[i].imshow(fmap, cmap='viridis')
# # # # #         axes[i].axis('off')
# # # # #         axes[i].set_title(f"Ch {i}")
    
# # # # #     plt.tight_layout()
# # # # #     plt.show()


# # # # # # Visualize each intermediate feature map
# # # # # for idx, feat in enumerate(intermediates):
# # # # #     if isinstance(feat, torch.Tensor):
# # # # #         name = f"Intermediate {idx}: shape={feat.shape}"
# # # # #         visualize_feature_map(feat, name=name)
# # # # #     else:
# # # # #         print(f"Intermediate {idx}: Not a tensor, skipping")


# # # # import torch
# # # # import matplotlib.pyplot as plt
# # # # import numpy as np
# # # # from hiera import hiera_base_224

# # # # # Load the pretrained model
# # # # model = hiera_base_224(pretrained=True).eval()

# # # # # Create a batch of dummy images
# # # # batch_images = torch.randn(16, 3, 224, 224)

# # # # # Global variable to store patch embedding
# # # # patch_embedding = None

# # # # # Hook to capture the patch embeddings after stem
# # # # def hook_patch_embed(module, input, output):
# # # #     global patch_embedding
# # # #     patch_embedding = output

# # # # # Register the hook
# # # # model.stem.register_forward_hook(hook_patch_embed)

# # # # # Forward pass with intermediate outputs
# # # # x, intermediates = model(batch_images, return_intermediates=True)

# # # # # Choose a sample index (e.g., the first image in the batch)
# # # # sample_idx = 0


# # # # def visualize_feature_map(feature_tensor, name="", max_channels=1):
# # # #     """Visualize up to `max_channels` channels from a feature map tensor."""
# # # #     if feature_tensor.dim() < 3:
# # # #         print(f"Skipping {name}: not enough dimensions (dim={feature_tensor.dim()})")
# # # #         return

# # # #     # If shape is [B, H, W], add channel dim to make it [B, 1, H, W]
# # # #     if feature_tensor.dim() == 3:
# # # #         feature_tensor = feature_tensor.unsqueeze(1)
# # # #     elif feature_tensor.dim() != 4:
# # # #         print(f"Skipping {name}: unsupported tensor shape {feature_tensor.shape}")
# # # #         return

# # # #     # Select the sample
# # # #     feature_maps = feature_tensor[sample_idx]  # shape: [C, H, W]
# # # #     num_channels = feature_maps.shape[0]

# # # #     # Adjust number of channels to visualize
# # # #     display_channels = min(max_channels, num_channels)

# # # #     fig, axes = plt.subplots(1, display_channels, figsize=(5 * display_channels, 5))
# # # #     if display_channels == 1:
# # # #         axes = [axes]  # Make iterable

# # # #     fig.suptitle(name, fontsize=14)

# # # #     for i in range(display_channels):
# # # #         fmap = feature_maps[i].detach().cpu().numpy()
# # # #         # Normalize for visualization
# # # #         fmap -= fmap.min()
# # # #         fmap /= fmap.max() + 1e-5
# # # #         axes[i].imshow(fmap, cmap='viridis')
# # # #         axes[i].axis('off')
# # # #         axes[i].set_title(f"Ch {i}")
    
# # # #     plt.tight_layout()
# # # #     plt.show()


# # # # # === Visualize Patch Embedding ===
# # # # if patch_embedding is not None:
# # # #     print(f"Patch Embedding Shape: {patch_embedding.shape}")
# # # #     visualize_feature_map(patch_embedding, name="Patch Embedding Output", max_channels=4)
# # # # else:
# # # #     print("Patch embedding hook failed or not triggered.")

# # # # # === Visualize Intermediate Layers ===
# # # # for idx, feat in enumerate(intermediates):
# # # #     if isinstance(feat, torch.Tensor):
# # # #         name = f"Intermediate {idx}: shape={feat.shape}"
# # # #         visualize_feature_map(feat, name=name, max_channels=2)
# # # #     else:
# # # #         print(f"Intermediate {idx}: Not a tensor, skipping")
# # # import torch
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from hiera import hiera_base_224

# # # # Load the pretrained model
# # # model = hiera_base_224(pretrained=True).eval()

# # # # Create a batch of dummy images
# # # batch_images = torch.randn(16, 3, 224, 224)

# # # # Global variable to store patch embedding
# # # patch_embedding = None

# # # # Hook to capture the patch embeddings from the first downsample layer
# # # def hook_patch_embed(module, input, output):
# # #     global patch_embedding
# # #     patch_embedding = output

# # # # Register the hook on the first downsample layer (stem equivalent)
# # # model.downsample_layers[0].register_forward_hook(hook_patch_embed)

# # # # Forward pass with intermediate outputs
# # # x, intermediates = model(batch_images, return_intermediates=True)

# # # # Choose a sample index
# # # sample_idx = 0

# # # def visualize_feature_map(feature_tensor, name="", max_channels=1):
# # #     """Visualize up to `max_channels` channels from a feature map tensor."""
# # #     if feature_tensor.dim() < 3:
# # #         print(f"Skipping {name}: not enough dimensions (dim={feature_tensor.dim()})")
# # #         return

# # #     # If shape is [B, H, W], add channel dim to make it [B, 1, H, W]
# # #     if feature_tensor.dim() == 3:
# # #         feature_tensor = feature_tensor.unsqueeze(1)
# # #     elif feature_tensor.dim() != 4:
# # #         print(f"Skipping {name}: unsupported tensor shape {feature_tensor.shape}")
# # #         return

# # #     # Select the sample
# # #     feature_maps = feature_tensor[sample_idx]  # shape: [C, H, W]
# # #     num_channels = feature_maps.shape[0]

# # #     # Adjust number of channels to visualize
# # #     display_channels = min(max_channels, num_channels)

# # #     fig, axes = plt.subplots(1, display_channels, figsize=(5 * display_channels, 5))
# # #     if display_channels == 1:
# # #         axes = [axes]  # Make iterable

# # #     fig.suptitle(name, fontsize=14)

# # #     for i in range(display_channels):
# # #         fmap = feature_maps[i].detach().cpu().numpy()
# # #         # Normalize for visualization
# # #         fmap -= fmap.min()
# # #         fmap /= fmap.max() + 1e-5
# # #         axes[i].imshow(fmap, cmap='viridis')
# # #         axes[i].axis('off')
# # #         axes[i].set_title(f"Ch {i}")
    
# # #     plt.tight_layout()
# # #     plt.show()

# # # # === Visualize Patch Embedding ===
# # # if patch_embedding is not None:
# # #     print(f"Patch Embedding Shape: {patch_embedding.shape}")
# # #     visualize_feature_map(patch_embedding, name="Patch Embedding Output", max_channels=4)
# # # else:
# # #     print("Patch embedding hook failed or not triggered.")

# # # # === Visualize Intermediate Layers ===
# # # for idx, feat in enumerate(intermediates):
# # #     if isinstance(feat, torch.Tensor):
# # #         name = f"Intermediate {idx}: shape={feat.shape}"
# # #         visualize_feature_map(feat, name=name, max_channels=2)
# # #     else:
# # #         print(f"Intermediate {idx}: Not a tensor, skipping")


# # # import torch
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from hiera import hiera_base_224

# # # # Load the pretrained model
# # # model = hiera_base_224(pretrained=True).eval()

# # # # Create a batch of dummy images
# # # batch_images = torch.randn(16, 3, 224, 224)

# # # # Global variable to store patch embedding
# # # patch_embedding = None

# # # # Hook to capture the patch embeddings from the first downsample layer
# # # def hook_patch_embed(module, input, output):
# # #     global patch_embedding
# # #     patch_embedding = output

# # # # Register the hook on the first downsample layer (stem equivalent)
# # # model.downsample_layers[0].register_forward_hook(hook_patch_embed)

# # # # Forward pass with intermediate outputs
# # # x, intermediates = model(batch_images, return_intermediates=True)

# # # # Choose a sample index
# # # sample_idx = 0

# # # def visualize_feature_map(feature_tensor, name="", max_channels=1):
# # #     """Visualize up to `max_channels` channels from a feature map tensor."""
# # #     if feature_tensor.dim() < 3:
# # #         print(f"Skipping {name}: not enough dimensions (dim={feature_tensor.dim()})")
# # #         return

# # #     # If shape is [B, H, W], add channel dim to make it [B, 1, H, W]
# # #     if feature_tensor.dim() == 3:
# # #         feature_tensor = feature_tensor.unsqueeze(1)
# # #     elif feature_tensor.dim() != 4:
# # #         print(f"Skipping {name}: unsupported tensor shape {feature_tensor.shape}")
# # #         return

# # #     # Select the sample
# # #     feature_maps = feature_tensor[sample_idx]  # shape: [C, H, W]
# # #     num_channels = feature_maps.shape[0]

# # #     # Adjust number of channels to visualize
# # #     display_channels = min(max_channels, num_channels)

# # #     fig, axes = plt.subplots(1, display_channels, figsize=(5 * display_channels, 5))
# # #     if display_channels == 1:
# # #         axes = [axes]  # Make iterable

# # #     fig.suptitle(name, fontsize=14)

# # #     for i in range(display_channels):
# # #         fmap = feature_maps[i].detach().cpu().numpy()
# # #         # Normalize for visualization
# # #         fmap -= fmap.min()
# # #         fmap /= fmap.max() + 1e-5
# # #         axes[i].imshow(fmap, cmap='viridis')
# # #         axes[i].axis('off')
# # #         axes[i].set_title(f"Ch {i}")
    
# # #     plt.tight_layout()
# # #     plt.show()

# # # # === Visualize Patch Embedding ===
# # # if patch_embedding is not None:
# # #     print(f"Patch Embedding Shape: {patch_embedding.shape}")
# # #     visualize_feature_map(patch_embedding, name="Patch Embedding Output", max_channels=4)
# # # else:
# # #     print("Patch embedding hook failed or not triggered.")

# # # # === Visualize Intermediate Layers ===
# # # for idx, feat in enumerate(intermediates):
# # #     if isinstance(feat, torch.Tensor):
# # #         name = f"Intermediate {idx}: shape={feat.shape}"
# # #         visualize_feature_map(feat, name=name, max_channels=2)
# # #     else:
# # #         print(f"Intermediate {idx}: Not a tensor, skipping")


# # import torch
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from hiera import hiera_base_224  # Make sure this import works from your local Hiera source

# # # Load the pretrained model and set to evaluation mode
# # model = hiera_base_224(pretrained=True).eval()

# # # Move model to the appropriate device (CPU or CUDA)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # # Create a batch of dummy images (e.g., batch of 16 RGB images, 224x224)
# # batch_images = torch.randn(16, 3, 224, 224).to(device)

# # # Global variable to store patch embedding
# # patch_embedding = None

# # # Hook to capture the patch embeddings from patch_embed layer
# # def hook_patch_embed(module, input, output):
# #     global patch_embedding
# #     patch_embedding = output

# # # Register the hook on the patch embedding layer
# # hook_handle = model.patch_embed.register_forward_hook(hook_patch_embed)

# # # Run a forward pass with intermediate outputs
# # x, intermediates = model(batch_images, return_intermediates=True)

# # # Choose a sample index for visualization
# # sample_idx = 0

# # def visualize_feature_map(feature_tensor, name="", max_channels=1):
# #     """Visualize up to `max_channels` channels from a feature map tensor."""
# #     if feature_tensor.dim() < 3:
# #         print(f"Skipping {name}: not enough dimensions (dim={feature_tensor.dim()})")
# #         return

# #     # Convert [B, H, W] → [B, 1, H, W]
# #     if feature_tensor.dim() == 3:
# #         feature_tensor = feature_tensor.unsqueeze(1)
# #     elif feature_tensor.dim() != 4:
# #         print(f"Skipping {name}: unsupported shape {feature_tensor.shape}")
# #         return

# #     # Select one sample from the batch
# #     feature_maps = feature_tensor[sample_idx]  # shape: [C, H, W]
# #     num_channels = feature_maps.shape[0]

# #     display_channels = min(max_channels, num_channels)
# #     fig, axes = plt.subplots(1, display_channels, figsize=(5 * display_channels, 5))
# #     if display_channels == 1:
# #         axes = [axes]

# #     fig.suptitle(name, fontsize=14)

# #     for i in range(display_channels):
# #         fmap = feature_maps[i].detach().cpu().numpy()
# #         # Normalize to [0, 1]
# #         fmap -= fmap.min()
# #         max_val = fmap.max()
# #         if max_val > 0:
# #             fmap /= max_val
# #         axes[i].imshow(fmap, cmap='viridis')
# #         axes[i].axis('off')
# #         axes[i].set_title(f"Channel {i}")
    
# #     plt.tight_layout()
# #     plt.show()

# # # === Visualize Patch Embedding ===
# # if patch_embedding is not None:
# #     print(f"Patch Embedding Shape: {patch_embedding.shape}")
# #     # patch_embedding shape: [B, N, C] → reshape to [B, C, H, W] for visualization
# #     try:
# #         B, N, C = patch_embedding.shape
# #         spatial_dim = int(N ** 0.5)
# #         patch_vis = patch_embedding.permute(0, 2, 1).reshape(B, C, spatial_dim, spatial_dim)
# #         visualize_feature_map(patch_vis, name="Patch Embedding Output", max_channels=1)
# #     except Exception as e:
# #         print(f"Error reshaping patch embedding for visualization: {e}")
# # else:
# #     print("Patch embedding hook failed or not triggered.")

# # # === Visualize Intermediate Layer Outputs ===
# # for idx, feat in enumerate(intermediates):
# #     if isinstance(feat, torch.Tensor):
# #         name = f"Intermediate {idx}: shape={feat.shape}"
# #         visualize_feature_map(feat, name=name, max_channels=1)
# #     else:
# #         print(f"Intermediate {idx}: Not a tensor, skipping")

# # # Optional: remove the hook to avoid side effects
# # hook_handle.remove()




# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# import numpy as np
# from PIL import Image
# from hiera import hiera_base_224

# # === Load a real image ===
# image = Image.open("your_image.jpg").convert("RGB")  # Replace with your image
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# # === Load the model ===
# model = hiera_base_224(pretrained=True).eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# image_tensor = image_tensor.to(device)

# # === Capture patch embedding ===
# patch_embedding = None
# def hook_patch_embed(module, input, output):
#     global patch_embedding
#     patch_embedding = output

# model.patch_embed.register_forward_hook(hook_patch_embed)

# # === Forward pass ===
# with torch.no_grad():
#     _, intermediates = model(image_tensor, return_intermediates=True)

# # === Helper: Convert tensor to displayable image ===
# def to_numpy_image(tensor):
#     img = tensor.detach().cpu().numpy()
#     img = np.transpose(img, (1, 2, 0))  # [H, W, C]
#     img = (img - img.min()) / (img.max() - img.min() + 1e-5)
#     return img

# # === Helper: Show grid of image patches and feature maps ===
# def show_patch_features(image_tensor, feature_tensor, stage_name, max_channels=1):
#     image_np = to_numpy_image(image_tensor[0])

#     # Get feature tensor shape [1, H, W, C]
#     H, W, C = feature_tensor.shape[1:]
#     features = feature_tensor[0].detach().cpu().numpy()  # [H, W, C]

#     # Optionally reduce channels (use PCA if C > 1)
#     if C > 1 and max_channels == 1:
#         pca = PCA(n_components=1)
#         features = pca.fit_transform(features.reshape(-1, C)).reshape(H, W)
#     else:
#         features = features[..., 0]

#     # Normalize feature map
#     features -= features.min()
#     features /= features.max() + 1e-5

#     # Resize original image to match grid
#     patch_size = 224 // H
#     img_resized = T.Resize((H * patch_size, W * patch_size))(image_tensor[0])
#     img_grid = to_numpy_image(img_resized)

#     # Display side-by-side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.imshow(img_grid)
#     ax1.set_title(f"{stage_name} - Image Grid ({H}x{W})")
#     ax1.axis('off')

#     ax2.imshow(features, cmap="viridis")
#     ax2.set_title(f"{stage_name} - Feature Map")
#     ax2.axis('off')
#     plt.tight_layout()
#     plt.show()

# # === Visualize patch embedding ===
# if patch_embedding is not None:
#     B, N, C = patch_embedding.shape
#     side = int(N ** 0.5)
#     patch_feat = patch_embedding.view(B, side, side, C)  # [B, H, W, C]
#     show_patch_features(image_tensor, patch_feat, stage_name="Patch Embedding")
# else:
#     print("No patch embedding captured")

# # === Visualize intermediate stages ===
# for idx, feat in enumerate(intermediates):
#     if isinstance(feat, torch.Tensor) and feat.ndim == 4:
#         show_patch_features(image_tensor, feat, stage_name=f"Intermediate Stage {idx}")


# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# import numpy as np
# from PIL import Image
# from hiera import hiera_base_224

# # === Load a real image ===
# image_path = "your_image.jpg"  # <-- Replace with your image file
# image = Image.open(image_path).convert("RGB")
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# # === Load the model ===
# model = hiera_base_224(pretrained=True).eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# image_tensor = image_tensor.to(device)

# # === Hook for patch embedding ===
# patch_embedding = None
# def hook_patch_embed(module, input, output):
#     global patch_embedding
#     patch_embedding = output

# model.patch_embed.register_forward_hook(hook_patch_embed)

# # === Forward pass with intermediate outputs ===
# with torch.no_grad():
#     _, intermediates = model(image_tensor, return_intermediates=True)

# # === Helper: Normalize and show single-channel grid with patch borders ===
# def show_feature_grid(feature_tensor, stage_name="Feature Map", use_pca=True):
#     """
#     feature_tensor: shape [1, H, W, C]
#     """
#     feat = feature_tensor[0].detach().cpu().numpy()  # [H, W, C]
#     H, W, C = feat.shape

#     # Reduce channels
#     if use_pca and C > 1:
#         flat = feat.reshape(-1, C)
#         reduced = PCA(n_components=1).fit_transform(flat).reshape(H, W)
#     else:
#         reduced = feat[..., 0]

#     # Normalize
#     reduced -= reduced.min()
#     reduced /= reduced.max() + 1e-5

#     # Plot with gridlines
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(reduced, cmap='viridis')
#     ax.set_title(f"{stage_name} ({H}×{W})")
#     for i in range(H + 1):
#         ax.axhline(i - 0.5, color='white', linewidth=0.5)
#     for j in range(W + 1):
#         ax.axvline(j - 0.5, color='white', linewidth=0.5)
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# # === Visualize Patch Embedding ===
# if patch_embedding is not None:
#     B, N, C = patch_embedding.shape
#     side = int(N ** 0.5)
#     patch_feat = patch_embedding.view(B, side, side, C)
#     show_feature_grid(patch_feat, stage_name="Patch Embedding")
# else:
#     print("Patch embedding was not captured.")

# # === Visualize Intermediate Stages ===
# for idx, feat in enumerate(intermediates):
#     if isinstance(feat, torch.Tensor) and feat.ndim == 4:
#         show_feature_grid(feat, stage_name=f"Intermediate Stage {idx}")


# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# import numpy as np
# from PIL import Image
# import matplotlib.patches as patches
# from hiera import hiera_base_224

# # === Load a real image ===
# image_path = "your_image.jpg"  # 🔁 Replace with your image file
# image = Image.open(image_path).convert("RGB")
# orig_size = image.size  # (W, H)

# # Resize for model input
# model_size = 224
# transform = T.Compose([
#     T.Resize((model_size, model_size)),
#     T.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# # === Load the model ===
# model = hiera_base_224(pretrained=True).eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# image_tensor = image_tensor.to(device)

# # === Hook for patch embedding ===
# patch_embedding = None
# def hook_patch_embed(module, input, output):
#     global patch_embedding
#     patch_embedding = output

# model.patch_embed.register_forward_hook(hook_patch_embed)

# # === Forward pass with intermediates ===
# with torch.no_grad():
#     _, intermediates = model(image_tensor, return_intermediates=True)

# # === Helper: Normalize and show single-channel grid with patch borders ===
# def show_feature_grid(feature_tensor, stage_name="Feature Map", use_pca=True):
#     feat = feature_tensor[0].detach().cpu().numpy()  # [H, W, C]
#     H, W, C = feat.shape

#     # Reduce channels
#     if use_pca and C > 1:
#         flat = feat.reshape(-1, C)
#         reduced = PCA(n_components=1).fit_transform(flat).reshape(H, W)
#     else:
#         reduced = feat[..., 0]

#     reduced -= reduced.min()
#     reduced /= reduced.max() + 1e-5

#     # Plot with gridlines
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(reduced, cmap='viridis')
#     ax.set_title(f"{stage_name} ({H}×{W})")
#     for i in range(H + 1):
#         ax.axhline(i - 0.5, color='white', linewidth=0.5)
#     for j in range(W + 1):
#         ax.axvline(j - 0.5, color='white', linewidth=0.5)
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# # === Helper: Draw token boxes on the original image ===
# def draw_token_boxes(image, grid_size, token_size, title="Token Regions"):
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.imshow(image)
#     ax.set_title(f"{title} ({grid_size[0]}x{grid_size[1]})")
#     for i in range(grid_size[0]):
#         for j in range(grid_size[1]):
#             y = i * token_size
#             x = j * token_size
#             rect = patches.Rectangle((x, y), token_size, token_size,
#                                      linewidth=0.5, edgecolor='lime', facecolor='none')
#             ax.add_patch(rect)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# # === Visualize Patch Embedding ===
# if patch_embedding is not None:
#     B, N, C = patch_embedding.shape
#     side = int(N ** 0.5)
#     patch_feat = patch_embedding.view(B, side, side, C)
#     show_feature_grid(patch_feat, stage_name="Patch Embedding")
#     draw_token_boxes(image.resize((224, 224)), grid_size=(56, 56), token_size=4, title="Patch Embedding Regions")
# else:
#     print("Patch embedding not captured.")

# # === Visualize Intermediate Stages ===
# stage_token_sizes = {
#     0: 8,   # Stage 1 → 28x28 grid → ~8x8 px per token
#     1: 16,  # Stage 2 → 14x14 grid
#     2: 32   # Stage 3 → 7x7 grid
# }

# for idx, feat in enumerate(intermediates):
#     if isinstance(feat, torch.Tensor) and feat.ndim == 4:
#         show_feature_grid(feat, stage_name=f"Intermediate Stage {idx}")
#         h, w = feat.shape[1:3]
#         token_px = stage_token_sizes.get(idx, 1)
#         draw_token_boxes(image.resize((224, 224)), grid_size=(h, w), token_size=token_px, title=f"Stage {idx} Token Regions")


# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# import numpy as np
# from PIL import Image
# import matplotlib.patches as patches
# from hiera import hiera_base_224

# # === Load image ===
# image_path = "your_image.jpg"  # Replace this
# image = Image.open(image_path).convert("RGB")
# orig_size = image.size
# model_size = 224
# transform = T.Compose([
#     T.Resize((model_size, model_size)),
#     T.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# # === Load model ===
# model = hiera_base_224(pretrained=True).eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# image_tensor = image_tensor.to(device)

# # === Hook for patch embedding ===
# patch_embedding = None
# def hook_patch_embed(module, input, output):
#     global patch_embedding
#     patch_embedding = output

# model.patch_embed.register_forward_hook(hook_patch_embed)

# # === Forward pass with intermediates ===
# with torch.no_grad():
#     _, intermediates = model(image_tensor, return_intermediates=True)

# # === Helper: extract PCA or channel 0
# def reduce_channels(feat, use_pca=True):
#     H, W, C = feat.shape
#     if use_pca and C > 1:
#         flat = feat.reshape(-1, C)
#         reduced = PCA(n_components=1).fit_transform(flat).reshape(H, W)
#     else:
#         reduced = feat[..., 0]
#     reduced -= reduced.min()
#     reduced /= reduced.max() + 1e-5
#     return reduced

# # === Helper: draw side-by-side original+grid and feature map ===
# def show_side_by_side(image, feature_tensor, stage_name, token_size):
#     img = image.resize((224, 224))
#     feat = feature_tensor[0].detach().cpu().numpy()  # [H, W, C]
#     H, W, C = feat.shape
#     feat_map = reduce_channels(feat)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     ax1.imshow(img)
#     ax1.set_title(f"{stage_name} — Tokens on Image")

#     # Draw token boxes
#     for i in range(H):
#         for j in range(W):
#             x = j * token_size
#             y = i * token_size
#             rect = patches.Rectangle((x, y), token_size, token_size,
#                                      linewidth=0.5, edgecolor='lime', facecolor='none')
#             ax1.add_patch(rect)

#     ax2.imshow(feat_map, cmap='viridis')
#     ax2.set_title(f"{stage_name} — Feature Map")
#     for i in range(H + 1):
#         ax2.axhline(i - 0.5, color='white', linewidth=0.5)
#     for j in range(W + 1):
#         ax2.axvline(j - 0.5, color='white', linewidth=0.5)

#     for ax in [ax1, ax2]:
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# # === Stage token sizes (in pixels, based on image 224×224)
# stage_token_sizes = {
#     "Patch Embedding": 4,
#     "Stage 1": 8,
#     "Stage 2": 16,
#     "Stage 3": 32
# }

# # === Visualize Patch Embedding ===
# if patch_embedding is not None:
#     B, N, C = patch_embedding.shape
#     side = int(N ** 0.5)
#     patch_feat = patch_embedding.view(B, side, side, C)
#     show_side_by_side(image, patch_feat, stage_name="Patch Embedding", token_size=stage_token_sizes["Patch Embedding"])
# else:
#     print("Patch embedding not captured.")

# # === Visualize Intermediate Stages ===
# for idx, feat in enumerate(intermediates):
#     if isinstance(feat, torch.Tensor) and feat.ndim == 4:
#         stage_name = f"Stage {idx + 1}"
#         token_px = stage_token_sizes.get(stage_name, 1)
#         show_side_by_side(image, feat, stage_name=stage_name, token_size=token_px)


import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from hiera import hiera_base_224

# === Load image ===
image_path = "your_image.jpg"  # 🔁 Replace with your image
image = Image.open(image_path).convert("RGB")
model_size = 224
transform = T.Compose([
    T.Resize((model_size, model_size)),
    T.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# === Load model ===
model = hiera_base_224(pretrained=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)

# === Hook for patch embedding ===
patch_embedding = None
def hook_patch_embed(module, input, output):
    global patch_embedding
    patch_embedding = output

model.patch_embed.register_forward_hook(hook_patch_embed)

# === Forward pass ===
with torch.no_grad():
    _, intermediates = model(image_tensor, return_intermediates=True)

# === Reduce channels for visualization
def reduce_channels(feat, use_pca=True):
    H, W, C = feat.shape
    if use_pca and C > 1:
        flat = feat.reshape(-1, C)
        reduced = PCA(n_components=1).fit_transform(flat).reshape(H, W)
    else:
        reduced = feat[..., 0]
    reduced -= reduced.min()
    reduced /= reduced.max() + 1e-5
    return reduced

# === Show original + feature map side by side
def show_side_by_side(image, feature_tensor, stage_name, token_size):
    img = image.resize((224, 224))
    feat = feature_tensor[0].detach().cpu().numpy()  # [H, W, C]
    H, W, C = feat.shape
    feat_map = reduce_channels(feat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{stage_name} — Tensor Shape: {feature_tensor.shape}", fontsize=12)

    ax1.imshow(img)
    ax1.set_title("Tokens on Image")
    for i in range(H):
        for j in range(W):
            x = j * token_size
            y = i * token_size
            rect = patches.Rectangle((x, y), token_size, token_size,
                                     linewidth=0.5, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)

    ax2.imshow(feat_map, cmap='viridis')
    ax2.set_title("Feature Map")
    for i in range(H + 1):
        ax2.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(W + 1):
        ax2.axvline(j - 0.5, color='white', linewidth=0.5)

    for ax in [ax1, ax2]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# === Token sizes (in pixels) for each stage
stage_token_sizes = {
    0: 4,   # Patch embedding (56x56)
    1: 8,   # Stage 1 (28x28)
    2: 16,  # Stage 2 (14x14)
    3: 32   # Stage 3 (7x7)
}

# === Visualize Patch Embedding (Stage 0)
if patch_embedding is not None:
    B, N, C = patch_embedding.shape
    side = int(N ** 0.5)
    patch_feat = patch_embedding.view(B, side, side, C)
    show_side_by_side(image, patch_feat, stage_name="Stage 0 (Patch Embedding)", token_size=stage_token_sizes[0])
else:
    print("Patch embedding not captured.")

# === Visualize Intermediate Stages (1–3)
for idx, feat in enumerate(intermediates):
    if isinstance(feat, torch.Tensor) and feat.ndim == 4:
        token_px = stage_token_sizes.get(idx , 1)
        show_side_by_side(image, feat, stage_name=f"Stage {idx + 1}", token_size=token_px)
