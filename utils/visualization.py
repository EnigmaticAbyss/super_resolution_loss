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

    # Create a figure
    num_rows = len(subfolders)
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, num_rows * 3))

    # Adjust spacing between rows
    fig.subplots_adjust(hspace=1)

    # Iterate through each subfolder
    for row_idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(parent_folder, subfolder)
        images = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Dynamically adjust the number of columns based on the number of images
        num_images = len(images)
        fig_row, fig_axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

        # Display each image in the row
        for col_idx, image_name in enumerate(images):
            image_path = os.path.join(subfolder_path, image_name)
            img = imread(image_path)

            ax = fig_axes[col_idx] if num_images > 1 else fig_axes
            ax.imshow(img)
            ax.set_title(image_name, fontsize=8)
            ax.axis('off')

        # Add row label with subfolder name
        fig_row.suptitle(subfolder, fontsize=12, y=1.05)

        # Hide the axes
        for ax in fig_axes if num_images > 1 else [fig_axes]:
            ax.axis('off')
        if row_idx==3:
            break    

    plt.show()



