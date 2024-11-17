
import sys
sys.path.append("C:/Users/Asus/Desktop/super_resolution_loss")



import os
import yaml
import torch as t
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models.esrgan import ESRGANGenerator
from torchsr.models import edsr
from losses.gradient_loss import GradientLoss
from utils.dataset import SRDataset
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from utils.train_utils import Trainer
import torch.nn as nn

class SuperResolutionTrainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._setup_paths()
        self._initialize_model()
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
        if model_type == "esrgan":
            self.model = ESRGANGenerator().to(self.device)
        elif model_type == "edsr":
             self.model = edsr(scale=4, pretrained=True).to(self.device)
            #  self.model = edsr(scale=4, pretrained=True)
        elif model_type == "NafNet":
            raise NotImplementedError("NafNet model not implemented yet!")
        elif model_type == "SwinIR":
            raise NotImplementedError("SwinIR model not implemented yet!")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _initialize_loss(self):
        # Initialize the loss function
        loss_type = self.config["loss_function"]
        if loss_type == "perceptual_loss":
            self.loss_fn = PerceptualLoss(layers=["relu_3"], device=self.device)
        elif loss_type == "lpips":
            self.loss_fn = LPIPSLoss(self.device)
        elif loss_type == "gradient_loss":
            self.loss_fn = GradientLoss(loss_type='l2', device=self.device)
        elif loss_type == "MSE_loss":
            self.loss_fn = nn.MSELoss(self.device)
        elif loss_type == "Frequency_loss":
            raise NotImplementedError("Frequency loss not implemented yet!")
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _initialize_data_loaders(self):
        # Initialize datasets and data loaders
        # 4x------>1
        self.train_dataset = SRDataset(
            hr_dir=self.config["train_hr_path"], 
            lr_dir=self.config["train_lr_path"], 
            transform=self.transform, 
            hr_size=(256, 256), 
            lr_size=(64, 64)
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True)

        self.valid_dataset = SRDataset(
            hr_dir=self.config["valid_hr_path"], 
            lr_dir=self.config["valid_lr_path"], 
            transform=self.transform, 
            hr_size=(256, 256), 
            lr_size=(64, 64)
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

    def train(self):
        # Start the training process
        epochs = self.config.get("epochs", 50)
        self.trainer.fit(epochs=epochs)

