import torch
from dataload.rsendloader import RSENDLoaderTrain, RSENDLoaderTest, SIDTestDataset, SIDTrainDataset
from torch.utils.data import DataLoader
from network.rsend_model import RSEND
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
from loss import VGGLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def compute_loss(model, low_light_imgs, well_lit_imgs, loss_components, low_output):
    low_output = model(low_light_imgs)
    loss_vgg = loss_components['criterion'](low_output, well_lit_imgs) #vgg loss
    return loss_vgg

def lr_schedule(epoch, warmup_epochs=25, max_lr_epochs=100, total_epochs=150, initial_lr = 1e-8, max_lr = 1e-5):
    if epoch < warmup_epochs:
        lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
    elif epoch < max_lr_epochs:
        lr = max_lr
    else:
        lr = max_lr * (1.0 - (epoch - max_lr_epochs) / (total_epochs - max_lr_epochs))
    return lr

def test(model, test_loader, device, save_dir, save=False, epoch=None):
    criterion = VGGLoss()
    val_loss = 0.0
    with torch.no_grad():
        for i, (low_light_imgs, well_lit_imgs) in enumerate(test_loader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output = model(low_light_imgs)
            loss = compute_loss(model, low_light_imgs, well_lit_imgs, {
                        'criterion': criterion,
                    }, low_output)
            val_loss += loss.item()
            if save:
                save_path = os.path.join(save_dir, f"val_epoch_{epoch}_batch_{i+1}.jpg")
                save_image(low_output, save_path, normalize=True)

    return val_loss

def train(model, train_loader, val_dataloader, device, save_dir, num_epochs=150):
    weights_dir_path = os.path.join("./weights", save_dir)
    if not os.path.exists(weights_dir_path):
        os.makedirs(weights_dir_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    criterion = VGGLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
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
        
        if (epoch + 1) % 30 == 0:
            val_loss = test(model, val_dataloader, device, save_dir, epoch=epoch, save=True)
        else:
            val_loss = test(model, val_dataloader, device, save_dir, epoch=epoch)    
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RSEND model for low light image enhancement."
    )
    parser.add_argument('--save_dir', type=str, default="", help='Directory to save the model and outputs')
    parser.add_argument('--train_dataset', type=str, default="", help='Training dataset directory or identifier')
    parser.add_argument('--val_dataset', type=str, default="", help="Validation dataset directory or identifier")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--epoch', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and validation")
    parser.add_argument('--state_dict', type=str, default="", help='Path to the state dict file')
    parser.add_argument('--train_SID', type=bool, default=False, help="Whether to train SID model")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize the model and load weights
    model = RSEND()
    state_dict = torch.load(args.state_dict, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    train_dataset = RSENDLoaderTrain(args.train_dataset)
    val_dataset = RSENDLoaderTest(args.val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Start training
    train(model, train_loader, val_loader, device, args.save_dir, num_epochs=args.epoch)
    

    