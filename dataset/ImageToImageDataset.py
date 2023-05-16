import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

class ImageToImageDataset(Dataset):
    def __init__(self, root, input_transform=None, label_transform=None):
        self.root = root
        self.input_transform = input_transform
        self.label_transform = label_transform

        self.input_dir = os.path.join(root, 'input')
        self.label_dir = os.path.join(root, 'label')

        self.input_filenames = sorted(os.listdir(self.input_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))

    def __getitem__(self, index):
        # Load the input and label images using PIL
        input_img = Image.open(os.path.join(self.input_dir, self.input_filenames[index])).convert('RGB')
        label_img = Image.open(os.path.join(self.label_dir, self.label_filenames[index]))

        # Apply the input and label transforms
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        if self.label_transform is not None:
            label_img = self.label_transform(label_img)

        # Return the preprocessed input and label images
        return input_img, label_img

    def __len__(self):
        return len(self.input_filenames)


class ImageToImageDatasetWithName(Dataset):
    def __init__(self, root, input_transform=None, label_transform=None):
        self.root = root
        self.input_transform = input_transform
        self.label_transform = label_transform

        self.input_dir = os.path.join(root, 'input')
        self.label_dir = os.path.join(root, 'label')

        self.input_filenames = sorted(os.listdir(self.input_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))

    def __getitem__(self, index):
        # Load the input and label images using PIL
        input_img = Image.open(os.path.join(self.input_dir, self.input_filenames[index])).convert('RGB')
        label_img = Image.open(os.path.join(self.label_dir, self.label_filenames[index]))

        # Apply the input and label transforms
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        if self.label_transform is not None:
            label_img = self.label_transform(label_img)

        # Get filename without extension
        filename = os.path.splitext(self.input_filenames[index])[0]

        # Return the preprocessed input and label images along with the filename
        return input_img, label_img, filename

    def __len__(self):
        return len(self.input_filenames)

