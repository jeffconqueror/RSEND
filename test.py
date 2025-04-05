import torch
from torchvision.utils import save_image
import os
from dataload.rsendloader import UnpairedLowLightLoader, RSENDLoaderTest
from torchvision import transforms
from torch.utils.data import DataLoader
from network.rsend_model import RSEND
import argparse

def test_real(model, dataloader, device, save_dir):
    model.eval() 
    with torch.no_grad():
        for i, low_light_imgs in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            low_output = model(low_light_imgs)
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_final.jpg")
            save_path2 = os.path.join(save_dir, f"test_batch_{i+1}_low_original.jpg")
            save_image(low_output, save_path1, normalize=True)
            save_image(low_light_imgs, save_path2, normalize=True)


def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output = model(low_light_imgs)
            # I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            # low_output = R_low*I_low_3
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_truth.jpg")
            save_path2 = os.path.join(save_dir, f"test_batch_{i+1}_low.jpg")
            save_image(low_output, save_path, normalize=True)
            save_image(well_lit_imgs, save_path1, normalize=True)
            save_image(low_light_imgs, save_path2, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RSEND model for low light image enhancement."
    )
    parser.add_argument('--save_dir', type=str,
                        default="./Test_images",
                        help='Directory to save the model and outputs')
    parser.add_argument('--test_dataset', type=str, default="/home/jingchl6/.local/RSEND_initial/Train_data/LOLv2/Synthetic/test/",
                        help="Validation dataset directory or identifier")
    parser.add_argument('--device', type=str,
                        default="cuda:0",
                        help='Device to use for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--state_dict', type=str, 
                        default="/home/jingchl6/.local/RSEND/weights/v2syn/model_epoch_660.pth", help='Path to the state dict file')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = RSEND()
    state_dict = torch.load(args.state_dict)  # Load the state dict
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)  # Load the trained weights
    model.to(device)
    test_dataset = RSENDLoaderTest(args.test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(model, test_dataloader, device, save_dir)
    #best v1: /home/jingchl6/.local/RSEND_initial/weights/train_prune/LOLv1_prune_Dsize_finetune/model_epoch_196.pth
    #best v2_real: /home/jingchl6/.local/RSEND_initial/weights/train_prune/LOLv2Real_prune_Dsize_onlyvgg/model_epoch_743.pth
    #best v2_syn: /home/jingchl6/.local/RSEND/weights/v2syn/model_epoch_660.pth