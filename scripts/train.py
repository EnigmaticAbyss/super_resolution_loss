# python folder handler
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")


import argparse
import os
import torch
from models.esrgan import ESRGANGenerator
from losses.gradient_loss import GradientLoss
# from models import esrgan, edsr  # Import additional models here
from utils.dataset import SRDataset
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from utils.train_utils import train_epoch, validate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run super-resolution training with dynamic configuration")
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
args = parser.parse_args()


# Set HR and LR crop or resize sizes
hr_size = (128, 128)  # Target high-res size
lr_size = (32, 32)    # Target low-res size for a 4x scale


# # Load the configuration file
# config_file = "config/config_experiment1.yaml"  # Change for each experiment
# with open(config_file) as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)


# Load configuration from the specified file
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ESRGANGenerator().to(device)

# Load the model
if config["model"] == "esrgan":
    model = ESRGANGenerator().to(device)
elif config["model"] == "edsr":
    model = edsr.EDSRGenerator().to(device)
else:
    raise ValueError("Unsupported model specified in config.")

# loss_fn = LPIPSLoss(device)


# Load the loss function
if config["loss_function"] == "perceptual_loss":
   
    loss_fn = PerceptualLoss(layers=["relu_3"], device=device)
elif config["loss_function"] == "lpips":
     loss_fn = LPIPSLoss(device)
elif config["loss_function"] == "gradient_loss":
     loss_fn = GradientLoss(loss_type='l1', device=device) # Default to L1 loss
elif config["loss_function"] == "gradient_loss":
     loss_fn = GradientLoss(loss_type='l1', device=device) # Default to L1 loss
elif config["loss_function"] == "gradient_loss":
     loss_fn = GradientLoss(loss_type='l1', device=device) # Default to L1 loss          
else:
    raise ValueError("Unsupported loss function specified in config.")




# Set up AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 1e-4))


# Scheduler setup: ReduceLROnPlateau reduces the learning rate if validation loss does not improve
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.get("scheduler_gamma", 0.1), patience=config.get("scheduler_patience", 5))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])

# TensorBoard writer
experiment_name = os.path.splitext(os.path.basename(args.config))[0]
writer = SummaryWriter(log_dir=f"logs/tensorboard/{experiment_name}")


# Prepare the datasets and data loaders
transform = transforms.Compose(
    [transforms.ToTensor()
     ])
train_dataset = SRDataset(
    hr_dir=config["train_hr_path"], 
    lr_dir=config["train_lr_path"], 
    transform=transform, 
    hr_size=hr_size,
    lr_size=lr_size
) 
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

valid_dataset = SRDataset(hr_dir=config["valid_hr_path"], lr_dir=config["valid_lr_path"], transform=transform
                             , hr_size=hr_size,
    lr_size=lr_size
    )
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# Training and validation loop
for epoch in range(config["epochs"]):
    print(f"Training Epoch {epoch + 1}/{config['epochs']}")
    train_epoch(model, train_loader, optimizer, loss_fn, writer, epoch, device)
    
    # Validate and log metrics
    val_loss, val_psnr, val_ssim = validate(model, valid_loader, loss_fn, device)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
    writer.add_scalar("Metrics/SSIM", val_ssim, epoch)

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    
    # Save model checkpoint
    os.makedirs("saved_models", exist_ok=True)
    checkpoint_path = f"saved_models/{experiment_name}_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
