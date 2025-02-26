import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import re

DATASET_PATH = "/work/TALC/enel645_2025w/garbage_data/"

# Defining dataset subdirectories
train_dir = os.path.join(DATASET_PATH, "CVPR_2024_dataset_Train")
val_dir = os.path.join(DATASET_PATH, "CVPR_2024_dataset_Val")
test_dir = os.path.join(DATASET_PATH, "CVPR_2024_dataset_Test")

# Defining ImageNet-like normalization
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # rotate +/- 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Map class names to integer labels
CLASS_LABELS = {"Black": 0, "Blue": 1, "Green": 2, "TTR": 3}

class GarbageDataset(Dataset):
    """
    Custom dataset class for the Garbage classification dataset.
    Loads images and extracts text descriptions from filenames.
    """

    def __init__(self, root_dir, transform=None):
        """
        Parameters:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # List to store image paths, text descriptions, and labels

        # Iterate over class directories
        for class_name, class_label in CLASS_LABELS.items():
            class_path = os.path.join(root_dir, class_name)

            # Skip if the directory does not exist
            if not os.path.isdir(class_path):
                continue

            # Process each image in the class folder
            for img_name in os.listdir(class_path):
                if img_name.endswith(".png"):
                    img_path = os.path.join(class_path, img_name)
                    text_description = self.extract_text_from_filename(img_name)
                    self.samples.append((img_path, text_description, class_label))

    def extract_text_from_filename(self, filename):
        """
        Extracts a meaningful text description from an image filename.
        - Removes underscores and numbers.
        - Converts to lowercase.
        """
        filename = os.path.splitext(filename)[0]  # Remove file extension (.png)
        filename = re.sub(r'_\d+$', '', filename)  # Remove trailing numbers
        filename = filename.replace("_", " ") # Replace underscores with spaces
        return filename.lower()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text_description, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(img)
        
        return image, text_description, torch.tensor(label, dtype=torch.long)
    
# Create training, validation, and test datasets
train_dataset = GarbageDataset(train_dir, transform=image_transforms["train"])
val_dataset = GarbageDataset(val_dir, transform=image_transforms["val"])
test_dataset = GarbageDataset(test_dir, transform=image_transforms["test"])

# Create training, validation, and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Quick test to check if the dataset loader works
if __name__ == "__main__":
    image, text, label = train_dataset[0]
    print("Image shape:", image.shape) # Should be (3, 224, 224)
    print("Extracted text:", text) # Should be a string
    print("Label:", label) # Should be a tensor with a single integer

