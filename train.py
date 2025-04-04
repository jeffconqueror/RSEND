import torch
from dataload.rsendloader import RsendTrainLoader, RsendTestLoader, UnpairedLowLightLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.rsend_model import RSEND
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
from loss import VGGLoss, CharbonnierLoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torchsummary import summary
import warnings
import cv2
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
from util.get_psnr_ssim import calculate_average_psnr_ssim

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.2)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.2)


def compute_loss(model, low_light_imgs, well_lit_imgs, loss_components, low_output):
    low_output = model(low_light_imgs)
    loss_vgg = loss_components['criterion'](low_output, well_lit_imgs) #vgg loss
    return loss_vgg

def lr_schedule(epoch, warmup_epochs=75, max_lr_epochs=600, total_epochs=750, initial_lr = 1e-8, max_lr = 2e-5):
    if epoch < warmup_epochs:
        lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
    elif epoch < max_lr_epochs:
        lr = max_lr
    else:
        lr = max_lr * (1.0 - (epoch - max_lr_epochs) / (total_epochs - max_lr_epochs))
    return lr

def test(model, test_loader, device, save_dir, scheduler = None, save=False, epoch=None):
    criterion = VGGLoss()
    val_loss = 0.0
    with torch.no_grad():
        for i, (low_light_imgs, well_lit_imgs) in enumerate(test_loader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output = model(low_light_imgs)
            
            # Calculate the different loss components
            loss = compute_loss(model, low_light_imgs, well_lit_imgs, {
                        'criterion': criterion,
                    }, low_output)
            
            # Aggregate the total loss
            # loss = loss_spa + loss_col + loss_exp + 0.02 * loss_vgg1 + loss_charon1
            val_loss += loss.item()
            if save:
                save_path = os.path.join(save_dir, f"val_epoch_{epoch}_batch_{i+1}.jpg")
                save_image(low_output, save_path, normalize=True)

    return val_loss

def train(model, train_loader, val_dataloader, device, save_dir, num_epochs=750):
    weights_dir_path = os.path.join("./weights", save_dir)
    if not os.path.exists(weights_dir_path):
        os.makedirs(weights_dir_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) 
    criterion = VGGLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            lr = lr_schedule(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            low_output = model(low_light_imgs)
            loss = compute_loss(model, low_light_imgs, well_lit_imgs, {
                'criterion': criterion,
            }, low_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        ##validation part
        model.eval()
        val_loss = 0
        
        if (epoch + 1) % 5 == 0:
            val_loss = test(None, model, val_dataloader, device, save_dir, epoch=epoch, save=True)
        else:
            val_loss = test(None, model, val_dataloader, device, save_dir, epoch=epoch)    
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_loss = train_loss / len(train_loader)
        plateau_scheduler.step(avg_val_loss)
        if isinstance(avg_train_loss, torch.Tensor):
            avg_train_loss = avg_train_loss.detach().cpu().item()
        if isinstance(avg_val_loss, torch.Tensor):
            avg_val_loss = avg_val_loss.detach().cpu().item()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss}, Validation Loss: {avg_val_loss}')

        scheduler.step()
        # if avg_val_loss < best_val_loss:
            # best_val_loss = avg_val_loss
        weights_file_path = os.path.join("./weights", save_dir, "model_epoch_{}.pth".format(epoch))
        torch.save(model.state_dict(), weights_file_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_save_path = os.path.join(save_dir, "training_validation_loss_plot.png")
    plt.savefig(plot_save_path)