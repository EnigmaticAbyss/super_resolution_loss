import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, val_dataloader, device='cuda',early_stopping_patience=10,writer=None):
        """
        Initializes the Trainer class.
        
        Parameters:
        - model: The model to be trained.
        - optimizer: The optimizer for training.
        - loss_fn: Loss function to be used during training.
        - train_dataloader: DataLoader for the training dataset.
        - val_dataloader: DataLoader for the validation dataset.
        - device: Device to use ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.early_stop_num = early_stopping_patience
        self.best_value_loss = float("inf")
        self.epochs_no_improve = 0
        self.writer= writer
    # Dynamically determine model and loss function names
        self.model_name = self.model.__class__.__name__
        self.loss_fn_name = self.loss_fn.__class__.__name__   

    def train_epoch(self, epoch):
        """
        Trains the model for one epoch.
        
        Parameters:
        - epoch: The current epoch number (used for logging).
        
        Returns:
        - avg_loss: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0

        for i, (lr_imgs, hr_imgs) in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
            lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
            # print("lr")
            # print(lr_imgs.shape)
            sr_imgs = self.model(lr_imgs)
            # print("HR img")
            # print(hr_imgs.shape)
            # print("SR IMG")
            # print(sr_imgs.shape)
            # Calculate loss
            loss = self.loss_fn(sr_imgs, hr_imgs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            # Optionally: log training loss for every batch (if you have a logging setup)
            # writer.add_scalar("Loss/train", loss.item(), epoch * len(self.train_dataloader) + i)

        avg_loss = total_loss / len(self.train_dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        return avg_loss

    def validate(self, epoch_counter):
        """
        Validates the model on the validation dataset.
        
        Returns:
        - avg_loss: The average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx,(lr_imgs, hr_imgs) in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                            
                lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                sr_imgs = self.model(lr_imgs)
       
                # Calculate loss
                loss = self.loss_fn(sr_imgs, hr_imgs)
  
                total_loss += loss.item()

                # Optionally: calculate additional metrics like PSNR and SSIM
                # psnr = calculate_psnr(sr_imgs, hr_imgs)
                # ssim = calculate_ssim(sr_imgs, hr_imgs)
                # log or accumulate these metrics if desired
            # Visualize feature maps every 5 epochs and on the first batch of validation
            if self.writer is not None and epoch_counter % 5 == 0 and batch_idx == 0:
                # Extract features for SR and HR images
                sr_features = self.loss_fn.extract_features(sr_imgs)
                hr_features = self.loss_fn.extract_features(hr_imgs)

                # Create grids for SR and HR features
                sr_grids = self.loss_fn.create_feature_grid(sr_features)
                hr_grids = self.loss_fn.create_feature_grid(hr_features)

                # Log the grids of SR and HR feature maps to TensorBoard
                for i, (sr_grid, hr_grid) in enumerate(zip(sr_grids, hr_grids)):
                    # Log feature maps
                    self.writer.add_image(f"FeatureMaps/SR/layer_{i}", sr_grid, epoch_counter)
                    self.writer.add_image(f"FeatureMaps/HR/layer_{i}", hr_grid, epoch_counter)

                    # Create side-by-side comparison and log it
                    side_by_side = torch.cat([sr_grid, hr_grid], dim=2)  # Concatenate along the width (horizontal)
                    self.writer.add_image(f"FeatureMaps/Comparison/layer_{i}", side_by_side, epoch_counter)

        avg_loss = total_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_loss}")
        return avg_loss

    def fit(self, epochs):
        """
        Trains the model for the specified number of epochs.
        
        Parameters:
        - epochs: The number of epochs to train for.
        
        Returns:
        - train_losses: List of average training losses for each epoch.
        - val_losses: List of average validation losses for each epoch.
        """
        assert self.early_stop_num > 0 or epochs > 0
        train_losses = []
        val_losses = []
        epoch_counter = 0
 
            
        #best model location
        best_model_path = f"saved_models/{self.model_name}_{self.loss_fn_name}.pth"
        temp_model_path = "saved_models"
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min= 1e-6)

        while True:
            epoch_counter += 1
            print(f"Starting Epoch {epoch_counter }/{epochs}")
            
            
            # Train and validate for one epoch
            train_loss = self.train_epoch(epoch_counter)
            val_loss = self.validate(epoch_counter)
            
            
            #  # Logging info
            self.writer.add_scalar("Loss/train", train_loss, epoch_counter)
            self.writer.add_scalar("Loss/validation", val_loss, epoch_counter)
        
            # Append losses to lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Step the cosine annealing scheduler
            scheduler.step()
            # find the best loss until now
            if val_loss < self.best_value_loss:
                self.epochs_no_improve = 0
                self.best_value_loss = val_loss
                # Optionally: Save the model checkpoint or log epoch metrics
                self.save_checkpoint(best_model_path,temp_model_path) # If checkpoint saving is required

            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs.")
            if   epoch_counter >= epochs:
                break
                   
            if  self.epochs_no_improve >= self.early_stop_num:
                print("Early stopping triggered. you have reached the limit of staying the same!")
                break



        return train_losses, val_losses








    def save_checkpoint(self,best_path,temp_path):
        """
        Saves the model checkpoint.
        
        Parameters:
        - best_path: location of the best model yet!
        """

        # Remove any other temporary model checkpoints in the directory with the same name format as experience
        for filename in os.listdir(temp_path):
            if filename.startswith(f"{self.model_name}_{self.loss_fn_name}") and filename.endswith(".pth"):
                os.remove(os.path.join(temp_path, filename))
        # Save the best model and delete any other checkpoints       
        torch.save(self.model.state_dict(), best_path)
        print(f"Training completed. Best model saved at {best_path} with validation loss {self.best_value_loss:.4f}")


