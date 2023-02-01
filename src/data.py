import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch


class RGBA2RGB(object):
    """
    Transforms a 4-channel image to a 3-channel image by removing the alpha channel. Sometimes necessary in the dataset.
    """
    def __call__(self, img):
        return img[:3]


class MontevideoDirtyContainerDataset(Dataset):
    """
    Custom dataset implementation for the Montevideo Dirty Container Dataset.
    """
    def __init__(self, metadata_csv, root_dir, transform=None, set="train"):
        self.annotations = pd.read_csv(metadata_csv, sep=";")
        self.root_dir = root_dir
        self.transform = transform
        self.set = set

        # convert labels to 0 and 1
        self.annotations['label'] = self.annotations['label'].map({'clean': 0, 'dirty': 1})

        if set == "train":
            self.annotations = self.annotations[self.annotations["split"] == "train"]
        elif set == "test":
            self.annotations = self.annotations[self.annotations["split"] == "test"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = read_image(img_path)
        label = self.annotations.iloc[index, 3]
        id = self.annotations.iloc[index, 0]

        if self.transform:
            image = self.transform(image)

        return image, label, id


def get_dataloaders(batch_size, root_dir, metadata_csv, split=0.8, use_augmentation=False):
    """
    Returns train, validation and test data loaders.

    :param batch_size: batch size
    :param root_dir: root directory of dataset
    :param metadata_csv: path to metadata csv
    :param split: split ratio for train and validation from dedicated "train" files
    :param use_augmentation: whether to use data augmentation
    :return: train, validation and test data loaders
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        RGBA2RGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data_augmentation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        RGBA2RGB(),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.25, 20), value=0, inplace=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = MontevideoDirtyContainerDataset(
        metadata_csv=metadata_csv, root_dir=root_dir, set="train", transform=train_data_augmentation_transforms if use_augmentation else transform
    )
    val_dataset = MontevideoDirtyContainerDataset(
        metadata_csv=metadata_csv, root_dir=root_dir, set="train", transform=transform
    )
    test_dataset = MontevideoDirtyContainerDataset(
        metadata_csv=metadata_csv, root_dir=root_dir, set="test", transform=transform
    )
    # split train_val_dataset into train and validation
    train_size = int(split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_sample_ids, val_sample_ids = torch.utils.data.random_split(list(range(len(train_dataset))), [train_size, val_size])
    train_set = torch.utils.data.Subset(train_dataset, train_sample_ids.indices)
    val_set = torch.utils.data.Subset(val_dataset, val_sample_ids.indices)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
