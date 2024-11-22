import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, subset, labels_file, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            subset (str): Subfolder for the dataset subset (e.g., train.X1, val.X).
            labels_file (str): Path to the labels JSON file.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform

        # Load the labels JSON
        with open(labels_file, 'r') as f:
            self.labels_map = json.load(f)  # {"n1": "label1", "n2": "label2", ...}

        # Dynamically create a mapping from n-prefixed labels to zero-based indices
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels_map.keys())}
        self.index_to_label = {idx: self.labels_map[label] for label, idx in self.label_to_index.items()}

        # Gather all image paths and corresponding zero-based numeric labels
        self.image_paths = []
        self.numeric_labels = []  # Store zero-based class indices like 0, 1, ...

        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                # Get the zero-based index for the class folder
                zero_based_label = self.label_to_index[class_folder]
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    self.image_paths.append(image_path)
                    self.numeric_labels.append(zero_based_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        numeric_label = self.numeric_labels[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")  # Ensure images are in RGB format

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(numeric_label).long()

    def get_label(self, idx):
        """
        Retrieve the human-readable label for a specific index.
        
        Args:
            idx (int): Index of the item in the dataset.
        
        Returns:
            str: Human-readable label for the given index.
        """
        if idx < 0 or idx >= len(self.numeric_labels):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self)}")
        
        numeric_label = self.numeric_labels[idx]
        return self.index_to_label[numeric_label]
