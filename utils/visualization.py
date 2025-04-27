

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread




def display_images_from_folders(parent_folder):
    """
    Display all images in each subfolder in a single row with the subfolder name as a label.
    
    :param parent_folder: The path to the parent folder containing subfolders with images.
    """
    # Get the list of subfolders
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    # Iterate through each subfolder
    for row_idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(parent_folder, subfolder)
        images = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Dynamically adjust the number of columns and figure size
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))  # Adjust width dynamically

        # Display each image in the row
        for col_idx, image_name in enumerate(images):
            image_path = os.path.join(subfolder_path, image_name)
            img = imread(image_path)
            
            
            # Remove "SRImage" (case-sensitive) from the title
            clean_name = image_name.replace("SRImage_", "")
            clean_name = clean_name.replace(".png", "")


            ax = axes[col_idx] if num_images > 1 else axes  # Handle single-image case
            ax.imshow(img)
            ax.set_title(clean_name, fontsize=5)
            ax.axis('off')

        # Add row label with subfolder name
        fig.suptitle(subfolder, fontsize=14, y=0.95)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)  # Reserve space for the title

        # Stop after a few rows if needed (e.g., to avoid overly large plots)
        if row_idx == 3:
            break

    plt.show()
