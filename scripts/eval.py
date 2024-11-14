# python folder handler
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")




import torch
from models.esrgan import ESRGANGenerator
from utils.dataset import SRDataset
from utils.train_utils import validate
from utils.visualization import display_comparison_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import argparse
import os

# config_files = glob.glob("config/config_experiment*.yml")

# parser = argparse.ArgumentParser(description="Run super-resolution training with dynamic configuration")
# parser.add_argument("--config", type=str, default=config_files, required=True, help="Path to the config file")
# args = parser.parse_args()
# Load configuration
with open("config_old/config_experiment2.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ESRGANGenerator().to(device)
model.load_state_dict(torch.load("saved_models/config_experiment2_epoch_20.pth"))  # Specify your saved model path
model.eval()

# Prepare validation dataset and loader
transform = transforms.Compose([transforms.ToTensor()])
valid_dataset = SRDataset(hr_dir=config["valid_hr_path"], lr_dir=config["valid_lr_path"], transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# Example coordinates for areas to highlight in relative coordinates (adjust as needed)
circles = [
    (0.5, 0.5, 0.1),  # Center circle with radius 0.1
    (0.7, 0.3, 0.08)  # Another point of interest
]

# Iterate through the validation set and display comparisons
with torch.no_grad():
    for lr_img, hr_img in valid_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        sr_img = model(lr_img)
    
    
        # Display a batch comparison (up to 5 images)
        display_comparison_batch(lr_img, sr_img, hr_img, circles=circles, max_examples=5)
        break  # Remove this if you want to display more than one batch
