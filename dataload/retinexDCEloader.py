import os
import torch
import torch.utils.data as data
from PIL import Image
import glob
import random
from torchvision import transforms


random.seed(1143)


def populate_low_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "low/*.png")
    train_list = image_list_lowlight
    return train_list

def populate_high_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "high/*.png")
    train_list = image_list_lowlight
    return train_list

	

class retinexDCE_loader_train(data.Dataset):
    def __init__(self, lowlight_images_path) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        # Pairing and shuffling
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = 224
        print("Total training examples:", len(self.paired_list))
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomCrop(self.size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    def __getitem__(self, index):
        data_lowlight_path, data_highlight_path = self.paired_list[index]
        data_lowlight = Image.open(data_lowlight_path)
        data_highlight = Image.open(data_highlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
        data_lowlight = self.transform(data_lowlight)
        data_highlight = data_highlight.resize((self.size,self.size), Image.LANCZOS)
        data_highlight = self.transform(data_highlight)
        
        return data_lowlight, data_highlight
       
    
    def __len__(self):
        return len(self.paired_list)
    
    
    
class retinexDCE_loader_test(data.Dataset):
    def __init__(self, lowlight_images_path, size=224) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        # Pairing and shuffling
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = size
        print("Total training examples:", len(self.paired_list))
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    def __getitem__(self, index):
        data_lowlight_path, data_highlight_path = self.paired_list[index]
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
        data_lowlight = self.transform(data_lowlight)
        data_highlight = Image.open(data_highlight_path)
        data_highlight = data_highlight.resize((self.size,self.size), Image.LANCZOS)
        data_highlight = self.transform(data_highlight)
        
        return data_lowlight, data_highlight
    
    
    def __len__(self):
        return len(self.paired_list)


class UnpairedLowLightLoader(data.Dataset):
    def __init__(self, images_path, size=224):
        """
        images_path: path to the directory containing low-light images.
        size: the desired size to resize the images.
        """
        self.images_path = images_path
        self.image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((self.size, self.size), Image.LANCZOS)
        image = self.transform(image)
        return image



