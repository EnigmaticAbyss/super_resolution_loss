import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def display_comparison_batch(lr_imgs, sr_imgs, hr_imgs, circles=None, max_examples=5):
    """
    Display low-resolution, super-resolution, and high-resolution images for a batch in a grid.
    Optionally circle specific areas of interest for quality comparison.
    
    Parameters:
    - lr_imgs: Batch of low-resolution images as a PyTorch tensor
    - sr_imgs: Batch of super-resolution images as a PyTorch tensor
    - hr_imgs: Batch of high-resolution images as a PyTorch tensor
    - circles: List of (x, y, radius) tuples to circle areas of interest (in relative coordinates)
    - max_examples: Maximum number of examples to display from the batch
    """
    def tensor_to_np(img):
        return img.cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy and reorder channels

    batch_size = min(lr_imgs.size(0), max_examples)  # Limit the number of images shown to max_examples
    
    fig, axs = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    # Ensure axs is always a 2D array, even if batch_size == 1
    if batch_size == 1:
        axs = np.expand_dims(axs, 0)
    
    for i in range(batch_size):
        lr_img_np = tensor_to_np(lr_imgs[i])
        sr_img_np = tensor_to_np(sr_imgs[i])
        hr_img_np = tensor_to_np(hr_imgs[i])
        
        # Display Low-Resolution Image
        axs[i, 0].imshow(lr_img_np)
        axs[i, 0].set_title("Low-Resolution")
        axs[i, 0].axis("off")
        
        # Display Super-Resolution Image
        axs[i, 1].imshow(sr_img_np)
        axs[i, 1].set_title("Super-Resolution")
        axs[i, 1].axis("off")
        
        # Display High-Resolution Image
        axs[i, 2].imshow(hr_img_np)
        axs[i, 2].set_title("High-Resolution")
        axs[i, 2].axis("off")
        
        # Add circles to highlight specific areas, if provided
        if circles:
            for ax in axs[i, :]:
                for (x, y, radius) in circles:
                    circ = patches.Circle((x * ax.get_xlim()[1], y * ax.get_ylim()[0]),
                                          radius=radius * min(ax.get_xlim()[1], ax.get_ylim()[0]),
                                          linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(circ)

    plt.tight_layout()
    plt.show()
