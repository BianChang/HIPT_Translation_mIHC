import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchmetrics import SSIM
from SwinVisionTranformer import SwinTransformer
import os
import numpy as np
from dataset.ImageToImageDataset import ImageToImageDataset
import argparse

# Define the search space
config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "lr_scheduler": tune.choice(["StepLR", "ExponentialLR", "CosineAnnealingLR"]),
    "batch_size": tune.choice([1, 2, 4]),
}


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create and configure the model using the provided hyperparameters
    batch_size = config["batch_size"]
    lr = config["lr"]
    model = SwinTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if config["lr_scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif config["lr_scheduler"] == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif config["lr_scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = nn.CrossEntropyLoss()

    # Load and preprocess the dataset using the provided patch_size
    # Define the transforms for the input and label images
    mean_data = [mean_r, mean_g, mean_b]  # replace with your calculated values
    std_data = [std_r, std_g, std_b]  # replace with your calculated values
    mean_label = [mean_ch1, mean_ch2, mean_ch3, mean_ch4]  # replace with your calculated values
    std_label = [std_ch1, std_ch2, std_ch3, std_ch4]  # replace with your calculated values

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_data, std=std_data)
    ])
    label_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_label, std=std_label)
    ])

    # Create instances of the ImageToImageDataset for the training, validation, and test sets
    parser = argparse.ArgumentParser(description='Train a Swin Transformer model for image-to-image translation')
    parser.add_argument('--train_path', type=str, default='/path/to/train', help='Path to training data')
    parser.add_argument('--val_path', type=str, default='/path/to/val', help='Path to validation data')
    parser.add_argument('--test_path', type=str, default='/path/to/test', help='Path to test data')
    args = parser.parse_args()

    train_dataset = ImageToImageDataset(args.train_path, input_transform=input_transform,
                                        label_transform=label_transform)
    val_dataset = ImageToImageDataset(args.val_path, input_transform=input_transform, label_transform=label_transform)
    test_dataset = ImageToImageDataset(args.test_path, input_transform=input_transform, label_transform=label_transform)

    # Create instances of the DataLoader for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train your model using the optimizer and scheduler
    for epoch in range(10):  # Loop over the dataset multiple times
        running_loss = 0.0
        epoch_ssim = []
        for i, (source, target) in enumerate(train_loader):
            inputs, labels = source.to(device), target.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            scheduler.step()
    # Evaluate your model and report the performance using tune.report()
        # validation loop
        model.eval()
        val_loss = 0.0
        val_ssim = []
        with torch.no_grad():
            for i, (source, target) in enumerate(val_loader):
                source, target = source.to(device), target.to(device)

                output = model(source)
                loss = criterion(output, target)
                val_loss += loss.item()

                # Calculate SSIM
                ssim_val = SSIM(output, target)
                val_ssim.append(ssim_val.cpu().numpy())
    running_loss /= len(train_loader)
    val_ssim = np.mean(val_ssim, axis=0)
    avg_val_ssim = np.mean(val_ssim)

    tune.report(loss=running_loss, accuracy=avg_val_ssim)

# Initialize Ray
ray.init()

# Define the experiment settings
num_samples = 50  # Number of samples for each hyperparameter set
max_num_epochs = 10  # Maximum number of epochs for training
gpus_per_trial = 1  # Number of GPUs per trial

# Run the experiment using ASHA (Asynchronous Successive Halving Algorithm) scheduler
asha_scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
search_alg = AxSearch(max_concurrent=2)  # You can use other search algorithms like BayesOptSearch, HyperOptSearch, etc.
search_alg = ConcurrencyLimiter(search_alg, max_concurrency=10)

analysis = tune.run(
    train_model,
    resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=asha_scheduler,
    search_alg=search_alg,
    metric="loss",
    mode="min",
)

# Get the best hyperparameters
best_trial = analysis.get_best_trial("loss", "min", "last")
best_config = best_trial.config
print("Best hyperparameters found were: ", best_config)
