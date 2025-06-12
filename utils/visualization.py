

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
import numpy as np



def display_images_from_folders(parent_folder, max_rows=4):
    """
    Display all images in each subfolder in a square grid with the subfolder name as a label.

    :param parent_folder: Path to the parent folder containing subfolders with images.
    :param max_rows: Maximum number of subfolders to display.
    """
    # Get sorted list of subfolders
    subfolders = sorted([f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))])

    for row_idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # Get sorted list of image files
        images = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not images:
            continue  # Skip empty folders

        num_images = len(images)
        grid_size = math.ceil(math.sqrt(num_images))  # Make square grid

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))

        # Flatten axes array for easy indexing
        if grid_size == 1:
            axes = [[axes]]
        elif grid_size > 1:
            axes = axes if isinstance(axes[0], (list, np.ndarray)) else [axes]

        for idx, image_name in enumerate(images):
            image_path = os.path.join(subfolder_path, image_name)
            img = imread(image_path)

            clean_name = image_name.replace("SRImage_", "")
            clean_name = os.path.splitext(clean_name)[0]

            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_title(clean_name, fontsize=6)
            ax.axis('off')

        # Hide any unused subplot axes
        for idx in range(num_images, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            axes[row][col].axis('off')

        fig.suptitle(subfolder, fontsize=14, y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

        if row_idx + 1 >= max_rows:
            break

    plt.show()
