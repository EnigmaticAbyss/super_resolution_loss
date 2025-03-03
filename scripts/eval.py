# python folder handler

import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")




import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.esrgan import ESRGANGenerator
from utils.dataset import SRDataset
from utils.metrics import calculate_psnr, calculate_ssim,calculate_fid_score,calculate_lpips_score
from torchsr.models import edsr
from torchsr.models import ninasr_b2

from models.SwinIR.models.network_swinir import SwinIR


class SuperResolutionEvaluator:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_loader = None
        self.writer = None
        self.iteration = 0
        
        self._initialize_model()
        self._load_model()
        self._prepare_test_loader()
        self._initialize_tensorboard()

    def _initialize_model(self):
        # Initialize the model
        model_type = self.config["model"]
        if model_type == "ESRGANGenerator":
            self.model = ESRGANGenerator().to(self.device)
        elif model_type == "EDSR":
         #  self.model = edsr(scale=4, pretrained=True)
            self.model = edsr(scale=4,pretrained=False).to(self.device)
        elif model_type == "NafNet":
            raise NotImplementedError("NafNet model not implemented yet!")
        elif model_type == "NinaSR":
            self.model = ninasr_b2(scale=4, pretrained=False).to(self.device)      
        elif model_type == "SwinIR":
        # Initialize SwinIR model for classical SR (e.g., x4 scaling)
            self.model = SwinIR(
                upscale=4, 
                in_chans=3, 
                img_size=56, 
                window_size=8, 
                img_range=224, 
                depths=[6, 6, 6, 6], 
                embed_dim=180, 
                num_heads=[6, 6, 6, 6], 
                mlp_ratio=2, 
                upsampler='pixelshuffle'
            ).to(self.device)   
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_model(self):
        # Load pre-trained model weights
        model_name = self.config["model"]
        loss_name = self.config["loss_function"]
        model_path = os.path.join(self.config.get("model_save_dir", "saved_models"), f"{model_name}_{loss_name}.pth")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.eval()

    def _prepare_test_loader(self):
        # Prepare the test dataset and DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        self.test_dataset = SRDataset(
            hr_dir=self.config["test_hr_path"], 
            lr_dir=self.config["test_lr_path"], 
            transform=transform, 
            hr_size=(256, 256), 
            lr_size=(64, 64)
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def _initialize_tensorboard(self):
        # Set up TensorBoard writer
        log_dir = self.config.get("log_dir", "logs/tensorboard")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir,f"experiment_{self.config['model']}_{self.config['loss_function']}"))
    def normalize_to_match_mean_std(self, sr_image, hr_image):
            """Normalize SR image to have the same mean and std as the HR image."""
            sr_mean = sr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
            sr_std = sr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel
            
            hr_mean = hr_image.mean(dim=[0, 2, 3], keepdim=True)  # Mean per channel
            hr_std = hr_image.std(dim=[0, 2, 3], keepdim=True)    # Std per channel

            # Normalize SR image to have same mean and std as HR image
            sr_image_normalized = (sr_image - sr_mean) / (sr_std + 1e-8)  # Normalize SR
            sr_image_normalized = sr_image_normalized * hr_std + hr_mean  # Match HR stats

            return sr_image_normalized
    def evaluate(self):
        output_dir = self.config.get("output_dir", "test_results")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the base directory exists

        with torch.no_grad():
            for lr_img, hr_img in self.test_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr_img = self.model(lr_img)
                sr_img = self.normalize_to_match_mean_std(sr_img,hr_img)

                # Calculate PSNR and SSIM
                single_psnr = calculate_psnr(sr_img, hr_img)
                single_ssim = calculate_ssim(sr_img, hr_img)
                # single_fid = calculate_fid_score(sr_img, hr_img)
                single_lpips = calculate_lpips_score(sr_img, hr_img)



                # Log metrics
                self.writer.add_scalar("Metrics/PSNR", single_psnr[0], self.iteration)
                self.writer.add_scalar("Metrics/SSIM", single_ssim[0], self.iteration)
                # self.writer.add_scalar("Metrics/FID", single_fid[0], self.iteration)
                self.writer.add_scalar("Metrics/LPIPS", single_lpips[0], self.iteration)
                

                # Create a folder for each image's results
                folder_name = f"image_{self.iteration}"
                folder_path = os.path.join(output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Define file paths
                lr_image_path = os.path.join(folder_path, "lr_image.png")
                hr_image_path = os.path.join(folder_path, "hr_image.png")
                sr_image_path = os.path.join(folder_path, f"SRImage_{self.config['model']}_{self.config['loss_function']}.png")

                # Save images
                save_image(lr_img, lr_image_path)
                save_image(hr_img, hr_image_path)
                save_image(sr_img, sr_image_path)

                self.iteration += 1

        print(f"Evaluation completed. Results saved in {output_dir}.")

