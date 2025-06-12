# python folder handler
import re
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")
from pathlib import Path




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

        self.model_type = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_loader = None
        self.writer = None
        self.iteration = 0

        if config_path.suffix == ".pth":
            # Extract filename without suffix
      
            filename = config_path.stem

            # Regex to extract parts
            match = re.match(r"([^_]+)_([^_]+)(?:_((?:\[\d+\])))?", filename)


            if match:
                self.model_type = match.group(1)
                self.loss_name = match.group(2)
                self.loss_layer = match.group(3)  # This will be None if not present

            else:
                raise ValueError("Filename format does not match expected pattern.")                        
        else:    
            with open(config_path, 'r') as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
                self.model_type = self.config["model"]            
                self.loss_name = self.config["loss_function"]
            
        self._initialize_model()
        self._load_model()
        self._prepare_test_loader()
        self._initialize_tensorboard()

    def _initialize_model(self):

        # Initialize the model
        # model_type = self.config["model"]
        if self.model_type == "ESRGANGenerator":
            self.model = ESRGANGenerator().to(self.device)
        elif self.model_type == "EDSR":
         #  self.model = edsr(scale=4, pretrained=True)
            self.model = edsr(scale=4,pretrained=False).to(self.device)
        elif self.model_type == "NafNet":
            raise NotImplementedError("NafNet model not implemented yet!")
        elif self.model_type == "NinaSR":
            self.model = ninasr_b2(scale=4, pretrained=False).to(self.device)      
        elif self.model_type == "SwinIR":
        # Initialize SwinIR model for classical SR (e.g., x4 scaling)
            self.model = SwinIR(
                upscale=4, 
                in_chans=3, 
                img_size=56, 
                window_size=8, 
                img_range=224, 
                depths=[6, 6, 6, 6], 
                embed_dim=180, 
                # num_heads=[6, 6, 6, 6], 
                mlp_ratio=2, 
                upsampler='pixelshuffle'
            ).to(self.device)      
 
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


    # def _load_model(self):
    #     # Load pre-trained model weights
    #     model_name = self.config["model"]
    #     loss_name = self.config["loss_function"]
    #     model_path = os.path.join(self.config.get("model_save_dir", "saved_models"), f"{model_name}_{loss_name}.pth")
        
    #     if os.path.exists(model_path):
    #         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    #         print(f"Model loaded successfully from {model_path}")
    #     else:
    #         raise FileNotFoundError(f"Model file not found: {model_path}")
        
    #     self.model.eval()
    def _load_model(self):
        # model_name = self.config["model"]
        # loss_name = self.config["loss_function"]
        main_dir = Path(__file__).resolve().parent.parent
        saved_model_dir = main_dir / "saved_models"

        if self.loss_layer:
            model_path =saved_model_dir / f"{self.model_type}_{self.loss_name}_{self.loss_layer}.pth"
        else:
            model_path = saved_model_dir / f"{self.model_type}_{self.loss_name}.pth"

            
        if model_path.exists():

            checkpoint = torch.load(model_path, map_location=self.device)  # Load model on the correct device
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)  # Move model to GPU after loading
            print(f"Model loaded successfully from {model_path}")
        else:

            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.eval()

    def _prepare_test_loader(self):
        # Prepare the test dataset and DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        self.test_dataset = SRDataset(
            hr_dir="data/DIV2K/test/HR", 
            lr_dir="data/DIV2K/test/LR_bicubic_X4", 
            transform=transform, 
            hr_size=(256, 256), 
            lr_size=(64, 64)
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def _initialize_tensorboard(self):
        # Set up TensorBoard writer
  
        main_dir = Path(__file__).resolve().parent.parent
        log_dir = main_dir / "logs/tensorboard"

        # print(log_dir)
        # print("THIIISSS")

        if self.loss_layer:
            experiment_name = f"experiment_{self.model_type}_{self.loss_name}_{self.loss_layer}"
        else:
            experiment_name = f"experiment_{self.model_type}_{self.loss_name}"
        self.writer = SummaryWriter(log_dir=log_dir / experiment_name)

    def normalize_to_match_mean_std(self, sr_image, hr_image):
        """Normalize SR image to have the same mean and std as the HR image."""
        sr_mean = sr_image.mean(dim=[0, 2, 3], keepdim=True)
        sr_std = sr_image.std(dim=[0, 2, 3], keepdim=True)
        
        hr_mean = hr_image.mean(dim=[0, 2, 3], keepdim=True)
        hr_std = hr_image.std(dim=[0, 2, 3], keepdim=True)

        sr_image_normalized = (sr_image - sr_mean) / (sr_std + 1e-8)
        sr_image_normalized = sr_image_normalized * hr_std + hr_mean

        return sr_image_normalized

    def evaluate(self):
        main_dir = Path(__file__).resolve().parent.parent
        output_dir = main_dir / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for lr_img, hr_img in self.test_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr_img = self.model(lr_img)
                sr_img = self.normalize_to_match_mean_std(sr_img, hr_img)

                single_psnr = calculate_psnr(sr_img, hr_img)
                single_ssim = calculate_ssim(sr_img, hr_img)
                single_lpips = calculate_lpips_score(sr_img, hr_img)

                self.writer.add_scalar("Metrics/PSNR", single_psnr[0], self.iteration)
                self.writer.add_scalar("Metrics/SSIM", single_ssim[0], self.iteration)
                self.writer.add_scalar("Metrics/LPIPS", single_lpips[0], self.iteration)

                folder_path = output_dir / f"image_{self.iteration}"
                folder_path.mkdir(parents=True, exist_ok=True)

                # Define file paths
                lr_image_path = folder_path / "lr_image.png"
                hr_image_path = folder_path / "hr_image.png"
                sr_image_path = folder_path / f"SRImage_{self.model_type}_{self.loss_name}_{self.loss_layer}.png"

                # Save images
                save_image(lr_img, lr_image_path)
                save_image(hr_img, hr_image_path)
                save_image(sr_img, sr_image_path)

                self.iteration += 1

        print(f"Evaluation completed. Results saved in {output_dir}.")