import torch
from torchvision.utils import save_image
import os
from dataload.rsendloader import UnpairedLowLightLoader
from torch.utils.data import DataLoader
from network.rsend_model import RSEND
import argparse

def test(model, dataloader, device, save_dir):
    model.eval() 
    with torch.no_grad():
        for i, low_light_imgs in enumerate(dataloader):
            print(f"Processing batch {i+1}/{len(dataloader)}")
            low_light_imgs = low_light_imgs.to(device)
            low_output = model(low_light_imgs)
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_enhanced.jpg")
            save_path2 = os.path.join(save_dir, f"test_batch_{i+1}_original.jpg")
            save_image(low_output, save_path1, normalize=True)
            save_image(low_light_imgs, save_path2, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RSEND model for low light image enhancement."
    )
    parser.add_argument('--save_dir', type=str, default="", help='Directory to save the model and outputs')
    parser.add_argument('--test_dataset', type=str, default="", help="Validation dataset directory or identifier")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--state_dict', type=str, default="", help='Path to the state dict file')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = RSEND()
    state_dict = torch.load(args.state_dict)  # Load the state dict
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict) 
    model.to(device)
    test_dataset = UnpairedLowLightLoader(args.test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test(model, test_dataloader, device, save_dir)
