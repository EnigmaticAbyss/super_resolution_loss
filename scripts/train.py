
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")

import argparse
import os
import torch
import torch.nn as nn

from models.esrgan import ESRGANGenerator
from torchsr.models import EDSR
from losses.gradient_loss import GradientLoss
from utils.dataset import SRDataset
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from utils.train_utils import train_epoch, validate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim.lr_scheduler import CosineAnnealingLR

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run super-resolution training with dynamic configuration")
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load configuration from the specified file
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
if config["model"] == "esrgan":
    model = ESRGANGenerator().to(device)
elif config["model"] == "edsr":
    model = EDSR(scale=4).to(device)
elif config["model"] == "NafNet":
    print("Not added yet!")
elif config["model"] == "SwinIR":
    print("Not added yet!")  
else:
    raise ValueError("Unsupported model specified in config.")

# Load the loss function
if config["loss_function"] == "perceptual_loss":
    loss_fn = PerceptualLoss(layers=["relu_3"], device=device)
elif config["loss_function"] == "lpips":
    loss_fn = LPIPSLoss(device)
elif config["loss_function"] == "gradient_loss":
    loss_fn = GradientLoss(loss_type='l1', device=device)
elif config["loss_function"] == "MSE_loss":    
    loss_fn = nn.MSELoss(device)
elif config["loss_function"] == "Frequency_loss":    
    print("Not yet implemented!")
else:
    raise ValueError("Unsupported loss function specified in config.")

# Set up optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 1e-4))
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.get("scheduler_gamma", 0.1), patience=config.get("scheduler_patience", 5))
scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config.get("min_lr", 1e-6))

# TensorBoard writer setup
model_name = config['model']
loss_name = config['loss_function']
writer = SummaryWriter(log_dir=f"logs/tensorboard/experiment_name_{model_name}_{loss_name}")

# Prepare the datasets and data loaders
# scale factor for X4 --> 4
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = SRDataset(hr_dir=config["train_hr_path"], lr_dir=config["train_lr_path"], transform=transform, hr_size=(256, 256), lr_size=(64, 64))
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
valid_dataset = SRDataset(hr_dir=config["valid_hr_path"], lr_dir=config["valid_lr_path"], transform=transform, hr_size=(256, 256), lr_size=(64, 64))
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# Define the fit function with early stopping and best model checkpointing you can add new one if you want
def fit(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, writer, config, early_stop_num=10):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = f"saved_models/{model_name}_{loss_name}.pth"
    temp_model_path = "saved_models"  # Temporary directory for interim checkpoints

    # Ensure the directory exists
    os.makedirs(temp_model_path, exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"Training Epoch {epoch + 1}/{config['epochs']}")
        train_loss= train_epoch(model, train_loader, optimizer, loss_fn, writer, epoch, device)
        
        # Validate and log metrics
        # val_loss, val_psnr, val_ssim = validate(model, valid_loader, loss_fn, device)
        val_loss= validate(model, valid_loader, loss_fn, device)
        # Logging info
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        
        #these two metrics for every epoch on validation set
        # writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
        # writer.add_scalar("Metrics/SSIM", val_ssim, epoch)

        # # Step the scheduler based on validation loss
        # scheduler.step(val_loss)
        # Step the cosine annealing scheduler
        scheduler.step()
        
        # Check for early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Remove any other temporary model checkpoints in the directory with the same name format as experience
            for filename in os.listdir(temp_model_path):
                if filename.startswith(f"{model_name}_{loss_name}") and filename.endswith(".pth"):
                    os.remove(os.path.join(temp_model_path, filename))
            # Save the best model and delete any other checkpoints
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with validation loss {best_val_loss:.4f}")
            
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # Early stopping
        if epochs_no_improve >= early_stop_num:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best model saved at {best_model_path} with validation loss {best_val_loss:.4f}")

# Run the fit function
fit(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, writer, config, early_stop_num=config.get("early_stop_num", 10))
