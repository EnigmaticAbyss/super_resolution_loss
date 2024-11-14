
from tqdm import tqdm
import torch
from utils.metrics import calculate_psnr, calculate_ssim

def train_epoch(model, dataloader, optimizer, loss_fn, writer, epoch, device):
    model.train()
    total_loss = 0
    
    # Wrap dataloader in tqdm for a progress bar
    for i, (lr_imgs, hr_imgs) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        sr_imgs = model(lr_imgs)

        # Calculate loss
        loss = loss_fn(sr_imgs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging training loss
        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + i)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
    return avg_loss

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    psnr_values = []
    ssim_values = []

    # Wrap dataloader in tqdm for a progress bar
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(dataloader, desc="Validation"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            # Calculate loss
            loss = loss_fn(sr_imgs, hr_imgs)
            total_loss += loss.item()

            # Calculate PSNR and SSIM for each image in the batch
            batch_psnr = calculate_psnr(sr_imgs, hr_imgs)
            batch_ssim = calculate_ssim(sr_imgs, hr_imgs)
            
            # Extend psnr_values and ssim_values with batch results
            psnr_values.extend(batch_psnr)
            ssim_values.extend(batch_ssim)

    # Calculate average metrics over all images in the validation set
    avg_loss = total_loss / len(dataloader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)

    print(f"Validation Loss: {avg_loss}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    return avg_loss, avg_psnr, avg_ssim


