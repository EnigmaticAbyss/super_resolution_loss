
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")



import os
import yaml
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import SRDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models.esrgan import ESRGANGenerator
from torchsr.models import edsr
from models.SwinIR.models.network_swinir import SwinIR

from losses.FrequencyLoss import FrequencyLoss
from losses.FourierFeaturePerceptualLoss import FourierFeaturePerceptualLoss
from losses.FourierDifferencePerceptualLoss import FourierDifferencePerceptualLoss
from losses.FourierPerceptualLoss import FourierPerceptualLoss
from losses.lpips_loss import LPIPSLoss
from losses.gradient_loss import GradientLoss
from losses.perceptual_loss import PerceptualLoss
from losses.perceptual_loss import PerceptualLoss
from losses.CombinedLoss import CombinedLoss
from losses.HieraPerceptual import HieraPerceptualLoss  # Import your custom loss class



from utils.train_utils import Trainer


class SuperResolutionTrainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
    # transforms.RandomCrop(size=(256, 256)),         # Random cropping
    # transforms.RandomHorizontalFlip(p=0.5),        # Horizontal flipping
    # transforms.RandomVerticalFlip(p=0.5),          # Vertical flipping
    # transforms.RandomRotation(degrees=5),          # Random rotation
    # transforms.ColorJitter(brightness=0.2,         # Color jitter
    #                        contrast=0.2,
    #                        saturation=0.2,
    #                        hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3),    # Gaussian blur
                            sigma=(0.1, 2.0)),
    transforms.ToTensor(),                         # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  std=[0.5, 0.5, 0.5])  # Normalize (optional)

        ]
    )
        self.model_name = self.config["model"]
        self.loss_fn_name = self.config["loss_function"]  
        self._setup_paths()
        self._initialize_model()
        self.load_pretrained_model()  # Load pretrained model right after model initialization

        self._initialize_loss()
        self._initialize_data_loaders()
        self._initialize_tensorboard()
        self._initialize_trainer()

     

    def _setup_paths(self):
        # Ensure required paths exist
        os.makedirs(self.config.get("log_dir", "logs/tensorboard"), exist_ok=True)
        os.makedirs(self.config.get("model_save_dir", "saved_models"), exist_ok=True)

    def _initialize_model(self):
        # Initialize the model
        model_type = self.config["model"]
        if model_type == "ESRGANGenerator":
            self.model = ESRGANGenerator().to(self.device)
        elif model_type == "EDSR":
             self.model = edsr(scale=4, pretrained=False).to(self.device)
            #  self.model = edsr(scale=4, pretrained=True)
        elif model_type == "NafNet":
            raise NotImplementedError("NafNet model not implemented yet!")
        elif model_type == "SwinIR":
        # Initialize SwinIR model for classical SR (e.g., x4 scaling)
            # raise NotImplementedError("NafNet model not implemented yet!")


<<<<<<< HEAD

=======
>>>>>>> 422f6bf21d5620e424544c210f5dbdda7dbe8089
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
            )

            # self.model = SwinIR(
            #     upscale=4, 
            #     in_chans=3, 
            #     img_size=64, 
            #     window_size=8, 
            #     img_range=255, 
            #     depths=[6, 6, 6, 6], 
            #     embed_dim=180, 
            #     num_heads=[6, 6, 6, 6], 
            #     mlp_ratio=2, 
            #     upsampler='pixelshuffle'
            # )
<<<<<<< HEAD
      
=======

>>>>>>> 422f6bf21d5620e424544c210f5dbdda7dbe8089
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _initialize_loss(self):
        # Initialize the loss function
        loss_type = self.config["loss_function"]
        if loss_type == "PerceptualLoss":
            self.loss_fn = PerceptualLoss(layers=["relu_3"], device=self.device)
        elif loss_type == "LPIPSLoss":
            self.loss_fn = LPIPSLoss(self.device)
        elif loss_type == "GradientLoss":
            self.loss_fn = GradientLoss(loss_type='l2').to(self.device)
        elif loss_type == "MSEloss":
            self.loss_fn = nn.MSELoss(self.device)
        elif loss_type == "Frequency_loss":
            self.loss_fn=  FrequencyLoss().to(self.device)
        elif loss_type == "FourierPerceptualLoss":
            self.loss_fn= FourierPerceptualLoss(PerceptualLoss(layers=["relu_3"], device=self.device)).to(self.device)
        elif loss_type == "FourierDifferencePerceptualLo":
            self.loss_fn= FourierDifferencePerceptualLoss(PerceptualLoss(layers=["relu_3"], device=self.device)).to(self.device)
        elif loss_type == "FourierFeaturePerceptualLoss":
            self.loss_fn=  FourierFeaturePerceptualLoss().to(self.device)
        elif loss_type == "CombinedLoss":
            self.loss_fn=  CombinedLoss(FrequencyLoss().to(self.device),PerceptualLoss(layers=["relu_3"], device=self.device)).to(self.device)                
        elif loss_type == "HieraPerceptualLoss":
           
            # Initialize Hiera-based perceptual loss
            self.loss_fn = HieraPerceptualLoss(layers=['2'], device='cuda')
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _initialize_data_loaders(self):
        # Initialize datasets and data loaders
        # 4x------>1
        self.train_dataset = SRDataset(
            hr_dir=self.config["train_hr_path"], 
            lr_dir=self.config["train_lr_path"], 
            transform=self.transform, 
            hr_size=(224, 224), 
            lr_size=(56, 56)
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True)

        self.valid_dataset = SRDataset(
            hr_dir=self.config["valid_hr_path"], 
            lr_dir=self.config["valid_lr_path"], 
            transform=self.transform, 
            hr_size=(224, 224), 
            lr_size=(56, 56)
        )
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config["batch_size"], shuffle=False)

    def _initialize_trainer(self):
        # Set up optimizer and scheduler
        optimizer_input = t.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config.get("weight_decay", 1e-4))
        # Initialize the training helper
        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer_input,  # Replace with your optimizer initialization
            loss_fn=self.loss_fn,
            train_dataloader=self.train_loader,
            val_dataloader=self.valid_loader,
            device=self.device,
            early_stopping_patience=self.config.get("early_stopping_patience", 10),writer=self.writer
            
        )

    def _initialize_tensorboard(self):
        # Set up TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.config.get("log_dir", "logs/tensorboard"),
                f"experiment_{self.config['model']}_{self.config['loss_function']}"
            )
        )
    def load_pretrained_model(self):
        """Check for a pretrained model in the saved directory and load it if available."""
        # save_dir = self.config.get("model_save_dir", "saved_models")
        save_dir = f"saved_models"

        model_name = f"{self.model_name}_{self.loss_fn_name}.pth"
        model_path = os.path.join(save_dir, model_name)
        model_path = os.path.join(self.config.get("model_save_dir", "saved_models"), f"{self.model_name}_{self.loss_fn_name}.pth")

        if os.path.exists(model_path):
            print(f"Pretrained model found at {model_path}. Loading...")
            checkpoint = t.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Pretrained model loaded successfully.")
        else:
            print(f"No pretrained model found at {model_path}. Starting from scratch.")


    def train(self):
        # Start the training process
        epochs = self.config.get("epochs", 50)
        self.trainer.fit(epochs=epochs)
