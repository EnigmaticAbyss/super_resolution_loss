# python folder handler

import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")

import torch
from models.esrgan import ESRGANGenerator
from utils.dataset import SRDataset
from utils.train_utils import validate

from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from torchsr.models import edsr
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_psnr, calculate_ssim
import torchvision.utils as vutils


# config_files = glob.glob("config/config_experiment*.yml")
print("small black")
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run super-resolution evaluation with dynamic configuration")
parser.add_argument("--eval", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load configuration from the specified file
with open(args.eval) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# TensorBoard writer setup
model_name = config['model']
loss_name = config['loss_function']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the trained model
if config["model"] == "esrgan":
    model = ESRGANGenerator().to(device)
elif config["model"] == "edsr":
    model = edsr(scale=4).to(device)
elif config["model"] == "NafNet":
    print("Not added yet!")
elif config["model"] == "SwinIR":
    print("Not added yet!")  
else:
    raise ValueError("Unsupported model specified in config.")




model.load_state_dict(torch.load(f"saved_models/{model_name}_{loss_name}.pth"))  # Specify your saved model path




model.eval()

# Prepare validation dataset and loader
transform = transforms.Compose([transforms.ToTensor()])
# test_dataset = SRDataset(hr_dir=config["valid_hr_path"], lr_dir=config["valid_lr_path"], transform=transform)
test_dataset = SRDataset(hr_dir="data/DIV2K/test/HR", lr_dir="data/DIV2K/test/LR_bicubic_X4", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Example coordinates for areas to highlight in relative coordinates (adjust as needed)
# circles = [
#     (0.5, 0.5, 0.1),  # Center circle with radius 0.1
#     (0.7, 0.3, 0.08)  # Another point of interest
# ]


# TensorBoard writer setup
model_name = config['model']
loss_name = config['loss_function']
writer = SummaryWriter(log_dir=f"logs/tensorboard/test_results_{model_name}_{loss_name}")

# Iterate through the validation set and display comparisons every part in loop is a batch which in test is one element

iteration = 0  # Initialize the iteration counter

output_dir = "test_results"  # Base directory to save images
os.makedirs(output_dir, exist_ok=True)  # Ensure the base directory exists

with torch.no_grad():
    for lr_img, hr_img in test_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        sr_img = model(lr_img)

        # Calculate PSNR and SSIM
        single_psnr = calculate_psnr(sr_img, hr_img)
        single_ssim = calculate_ssim(sr_img, hr_img)
   
        print(type(single_psnr))
        # Log the PSNR and SSIM metrics
        writer.add_scalar("Metrics/PSNR", single_psnr[0], iteration)
        writer.add_scalar("Metrics/SSIM", single_ssim[0], iteration)

        # Create a folder for this image
        folder_name = f"image_{iteration}"
        folder_path = os.path.join(output_dir, folder_name)

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder {folder_path} created.")
        else:
            print(f"Folder {folder_path} already exists.")
            
        # Define file paths
        lr_image_path = os.path.join(folder_path, "lr_image.png")
        hr_image_path = os.path.join(folder_path, "hr_image.png")
        sr_image_path = os.path.join(folder_path, f"SRImage_{model_name}_{loss_name}.png")

        # Save images only if they don't already exist
        if not os.path.exists(lr_image_path):
            vutils.save_image(lr_img, lr_image_path)
        if not os.path.exists(hr_image_path):
            vutils.save_image(hr_img, hr_image_path)
        if not os.path.exists(sr_image_path):
            vutils.save_image(sr_img, sr_image_path)
            
        iteration += 1  # Increment the iteration counter





