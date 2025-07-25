import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils  # ADD THIS for grids
import torchvision.transforms as T
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

# --- Helper functions ---

def tensor_to_image_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a [C,H,W] torch tensor (values ~[0,1]) to a numpy RGB image array [H,W,3] in [0,1].
    """
    img = tensor.detach().cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img

def reduce_channels(feat, use_pca=True):
    H, W, C = feat.shape
    # print("fek konm")
    # print(feat.shape)
    if use_pca and C > 1:
        flat = feat.reshape(-1, C)
        reduced = PCA(n_components=1).fit_transform(flat).reshape(H, W)
    else:
        reduced = feat[..., 0]
    reduced -= reduced.min()
    reduced /= reduced.max() + 1e-5
    return reduced

def fig_to_tensor(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    plt.close(fig)
    return torch.tensor(image).permute(2, 0, 1).float()

def show_hr_sr_features(hr_img, sr_img, hr_feat, sr_feat, stage_name, token_size):
    # print("stage1")
    # print(len(hr_feat))
    feat_hr = hr_feat[0].detach().cpu().numpy()
    feat_sr = sr_feat[0].detach().cpu().numpy()
    # print("stager")
    # print(feat_hr.shape)
    B, H, W, C = feat_hr.shape
    # print(feat_hr.shape)
    # print("stageraaaaaa")
    fmap_hr = reduce_channels(feat_hr[0])
    fmap_sr = reduce_channels(feat_sr[0])
    # print("stage2")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"{stage_name} — Shape: {hr_feat[0].shape}", fontsize=10)
    # print("stage3")

    # for i in range(H):
    #     for j in range(W):
    #         x = j * token_size
    #         y = i * token_size
    #         # one rectangle per axes
    #         rect_hr = patches.Rectangle((x, y), token_size, token_size,
    #                                     linewidth=0.5, edgecolor='lime', facecolor='none')
    #         rect_sr = patches.Rectangle((x, y), token_size, token_size,
    #                                     linewidth=0.5, edgecolor='lime', facecolor='none')

    #         axes[0][0].add_patch(rect_hr)
    #         axes[1][0].add_patch(rect_sr)



    axes[0][0].imshow(hr_img)
    axes[0][0].set_title("HR Image + Grid")
    axes[0][0].axis('off')

    axes[1][0].imshow(sr_img)
    axes[1][0].set_title("SR Image + Grid")
    axes[1][0].axis('off')

    axes[0][1].imshow(fmap_hr, cmap='viridis')
    axes[0][1].set_title("HR Feature Map")
    axes[0][1].axis('off')

    axes[1][1].imshow(fmap_sr, cmap='viridis')
    axes[1][1].set_title("SR Feature Map")
    axes[1][1].axis('off')
    # print("stage5")

    plt.tight_layout()
    # print("stage6")

    return fig_to_tensor(fig)


def apply_colormap_to_tensor(tensor_img, cmap_name='jet'):
    """
    Apply matplotlib colormap to a single-channel tensor image.

    Args:
    - tensor_img: torch.Tensor (1, H, W), values in [0,1]
    - cmap_name: matplotlib colormap name (default 'jet')

    Returns:
    - colored_tensor: torch.Tensor (3, H, W) with colormap applied
    """
    np_img = tensor_img.squeeze(0).cpu().numpy()  # (H, W)
    cmap = plt.get_cmap(cmap_name)
    colored_img = cmap(np_img)[:, :, :3]  # Drop alpha, shape (H, W, 3)
    colored_img = colored_img.transpose(2, 0, 1)  # (3, H, W)
    return torch.from_numpy(colored_img).float()


def add_border(tensor_img, border_size=3, border_color=(1,1,1)):
    """
    Add colored border to a 3-channel image tensor.
    
    Args:
    - tensor_img: torch.Tensor (3, H, W), assumed normalized [0,1]
    - border_size: int, thickness of border in pixels
    - border_color: tuple of 3 floats, RGB color of border in [0,1]

    Returns:
    - tensor_img_with_border: torch.Tensor (3, H+2*border_size, W+2*border_size)
    """
    c, h, w = tensor_img.shape
    # Create border tensor filled with border_color
    border = torch.ones((c, h + 2*border_size, w + 2*border_size), device=tensor_img.device) * torch.tensor(border_color, device=tensor_img.device).view(c,1,1)
    # Place original image in the center
    border[:, border_size:border_size+h, border_size:border_size+w] = tensor_img
    return border


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
        if hasattr(self.loss_fn, 'selected_layers'):
            self.loss_layer = self.loss_fn.selected_layers
        else:
            self.loss_layer = None  # Or handle it some other way
            
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
        # print("OUTSIDE")
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
                # print(batch_idx)
                # print("OUTSIDE")

                # and epoch_counter % 5 == 0
                if self.writer is not None and epoch_counter % 5 == 0 and batch_idx == 0:  
                # if self.writer is not None and  batch_idx == 0:                   
                 
                    if self.loss_fn_name == "HieraNoFreqPercepNoMSE":
                        sr_features = self.loss_fn.compute_features(sr_imgs)
                        hr_features = self.loss_fn.compute_features(hr_imgs)
                    else:
                        sr_features = self.loss_fn.extract_features(sr_imgs)
                        hr_features = self.loss_fn.extract_features(hr_imgs)

                    # sr_grids = self.loss_fn.create_feature_grid(sr_features)
                    # hr_grids = self.loss_fn.create_feature_grid(hr_features)

                    # normalized_hr = (hr_imgs[0] - hr_imgs[0].min()) / (hr_imgs[0].max() - hr_imgs[0].min() + 1e-5)

                    # border_size = 3  # pixels

                    # for i, (sr_grid, hr_grid) in enumerate(zip(sr_grids, hr_grids)):
                    #     # Add colored borders
                    #     sr_bordered = add_border(sr_grid, border_size, border_color=(0,1,0))    # Green border for SR
                    #     hr_bordered = add_border(hr_grid, border_size, border_color=(0,0,1))    # Blue border for HR

                    #     side_by_side = torch.cat([sr_bordered, hr_bordered], dim=2)  # Horizontally

                    #     # Diff map
                    #     diff = torch.abs(sr_grid - hr_grid)
                    #     diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
                    #     diff_gray = diff.mean(dim=0, keepdim=True)
                    #     diff_colored = apply_colormap_to_tensor(diff_gray, cmap_name='jet')
                    #     diff_bordered = add_border(diff_colored, border_size, border_color=(1,0,0))  # Red border for Diff

                        # self.writer.add_image(f"FeatureMaps/SR/layer_{i}", sr_bordered, epoch_counter)
                        # self.writer.add_image(f"FeatureMaps/HR/layer_{i}", hr_bordered, epoch_counter)
                        # self.writer.add_image(f"FeatureMaps/Comparison/layer_{i}", side_by_side, epoch_counter)
                        # self.writer.add_image(f"FeatureMaps/Diff_Colored/layer_{i}", diff_bordered, epoch_counter)
                        # self.writer.add_image(f"FeatureMaps/HR_Image/layer_{i}", normalized_hr, epoch_counter)

                    if self.loss_fn_name == "HieraNoFreqPercepNoMSE":
                        # hiera = self.loss_fn.hiera_model.eval().to(self.device)
                        stage_token_sizes = {0: 4, 1: 8, 2: 16, 3: 32}
              
                        hr_img = hr_imgs[0]
                        sr_img = sr_imgs[0]
                        
                        # pil_hr = T.ToPILImage()(hr_img.cpu()).resize((224, 224))
                        # pil_sr = T.ToPILImage()(sr_img.cpu()).resize((224, 224))
                        hr_np = tensor_to_image_array(hr_img)
                        sr_np = tensor_to_image_array(sr_img)                     
                     
                        # hr_np = tensor_to_image_array(hr_resized)
                        # sr_np = tensor_to_image_array(sr_resized)
                       
                     
                        # hr_feats = self.loss_fn.extract_features(hr_img)
                        # sr_feats = self.loss_fn.extract_features(sr_img)

                    # for i in self.loss_layer:
                        token_size = stage_token_sizes.get(int(self.loss_layer[0]), 4)
                        # print("before")
                        vis_tensor = show_hr_sr_features(
                            hr_np, sr_np, hr_features, sr_features, f"Stage {int(self.loss_layer[0])}", token_size
                        )
                        # print("tensor added")
                        # print(vis_tensor)
                        self.writer.add_image(f"HieraCompare/Stage_{int(self.loss_layer[0])}", vis_tensor, epoch_counter)

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
        best_model_path = f"saved_models/{self.model_name}_{self.loss_fn_name}_{self.loss_layer}.pth"
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
            if filename.startswith(f"{self.model_name}_{self.loss_fn_name}_{self.loss_layer}") and filename.endswith(".pth"):
                os.remove(os.path.join(temp_path, filename))
        # Save the best model and delete any other checkpoints       
        torch.save(self.model.state_dict(), best_path)
        print(f"Training completed. Best model saved at {best_path} with validation loss {self.best_value_loss:.4f}")


