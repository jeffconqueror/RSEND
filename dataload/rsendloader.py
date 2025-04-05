import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from torchvision import transforms
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(1143)


def populate_low_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "low/*.png")
    train_list = image_list_lowlight
    return train_list

def populate_high_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "high/*.png")
    train_list = image_list_lowlight
    return train_list

	

class RSENDLoaderTrain(data.Dataset):
    def __init__(self, lowlight_images_path) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = 224
        print("Image numbers: ", len(self.paired_list))
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
    
    
    
class RSENDLoaderTest(data.Dataset):
    def __init__(self, lowlight_images_path, size=224) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        
        # Pairing and shuffling
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = size
        print("Image numbers: ", len(self.paired_list))
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

#dataset for SID 
def populate_file_list(data_dir):
    """Helper function to populate file lists for .npy files, recursively."""
    file_list = sorted(glob.glob(os.path.join(data_dir, '**/*.npy'), recursive=True))
    print(f"Found {len(file_list)} files in {data_dir}")
    return file_list


def pair_files(short_list, long_list):
    """Pairs each long exposure file with all corresponding short exposure files in the same folder."""
    paired_list = []

    # Group short files by folder
    short_files_by_folder = {}
    for short_file in short_list:
        folder_name = os.path.basename(os.path.dirname(short_file))
        if folder_name not in short_files_by_folder:
            short_files_by_folder[folder_name] = []
        short_files_by_folder[folder_name].append(short_file)

    for long_file in long_list:
        # Match long file by folder name
        long_folder = os.path.basename(os.path.dirname(long_file))
        if long_folder in short_files_by_folder:
            matching_shorts = short_files_by_folder[long_folder]
            for short_file in matching_shorts:
                paired_list.append((short_file, long_file))
        else:
            print(f"Warning: No short files found for long exposure file in folder {long_folder}")

    return paired_list


class SIDTrainDataset(data.Dataset):
    def __init__(self, short_dir, long_dir, scale_factor=0.5):
        """
        Training dataset for SID with data augmentation.

        :param short_dir: Path to directory containing short-exposure .npy files.
        :param long_dir: Path to directory containing long-exposure .npy files.
        :param scale_factor: Factor by which to scale the images (e.g., 0.7 for 70%).
        """
        short_list = populate_file_list(short_dir)
        long_list = populate_file_list(long_dir)

        # Pair short and long files
        self.paired_list = pair_files(short_list, long_list)

        self.scale_factor = scale_factor
        print(f"Loaded {len(self.paired_list)} training examples.")

        # Transformations with data augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        short_path, long_path = self.paired_list[index]

        # Load short and long exposure images from .npy files
        short_image = np.load(short_path).astype(np.float32)
        long_image = np.load(long_path).astype(np.float32)

        #normalize to [0, 1]
        short_image = (short_image - short_image.min()) / (short_image.max() - short_image.min())
        long_image = (long_image - long_image.min()) / (long_image.max() - long_image.min())

        if len(short_image.shape) == 2:
            short_image = np.expand_dims(short_image, axis=-1)
        if len(long_image.shape) == 2:
            long_image = np.expand_dims(long_image, axis=-1)

        #convert to uint8
        short_image = (short_image * 255).astype(np.uint8)
        long_image = (long_image * 255).astype(np.uint8)
        short_image = cv2.resize(
            short_image, 
            (int(short_image.shape[1] * self.scale_factor), int(short_image.shape[0] * self.scale_factor))
        )
        long_image = cv2.resize(
            long_image, 
            (int(long_image.shape[1] * self.scale_factor), int(long_image.shape[0] * self.scale_factor))
        )

        # Apply transformations
        short_tensor = self.transform(short_image)
        long_tensor = self.transform(long_image)

        return short_tensor, long_tensor

    def __len__(self):
        return len(self.paired_list)


class SIDTestDataset(data.Dataset):
    def __init__(self, short_dir, long_dir, size=224):
        """
        Testing dataset for SID without data augmentation.

        :param short_dir: Path to directory containing short-exposure .npy files.
        :param long_dir: Path to directory containing long-exposure .npy files.
        :param size: Image resize dimensions.
        """
        short_list = populate_file_list(short_dir)
        long_list = populate_file_list(long_dir)

        #pair
        self.paired_list = pair_files(short_list, long_list)

        self.size = size
        print(f"Loaded {len(self.paired_list)} testing examples.")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        short_path, long_path = self.paired_list[index]

        short_image = np.load(short_path).astype(np.float32)
        long_image = np.load(long_path).astype(np.float32)

        #normalize to [0, 1]
        short_image = (short_image - short_image.min()) / (short_image.max() - short_image.min())
        long_image = (long_image - long_image.min()) / (long_image.max() - long_image.min())

        short_image = (short_image * 255).astype(np.uint8)
        long_image = (long_image * 255).astype(np.uint8)

        short_tensor = self.transform(short_image)
        long_tensor = self.transform(long_image)

        return short_tensor, long_tensor

    def __len__(self):
        return len(self.paired_list)
    
if __name__ == "__main__":
    short_dir_train = '/home/jingchl6/.local/RSEND_initial/Train_data/sid_processed/train/short_sid2'
    long_dir_train = '/home/jingchl6/.local/RSEND_initial/Train_data/sid_processed/train/long_sid2'

    short_dir_test = '/home/jingchl6/.local/RSEND_initial/Train_data/sid_processed/test/short_sid2'
    long_dir_test = '/home/jingchl6/.local/RSEND_initial/Train_data/sid_processed/test/long_sid2'

    #Initialize datasets
    train_dataset = SIDTrainDataset(short_dir_train, long_dir_train)
    test_dataset = SIDTestDataset(short_dir_test, long_dir_test)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
