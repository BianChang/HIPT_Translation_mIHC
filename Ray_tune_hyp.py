import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchmetrics import StructuralSimilarityIndexMeasure
from SwinVisionTranformer import SwinTransformer
import os
import numpy as np
from dataset.ImageToImageDataset import ImageToImageDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


SSIM = StructuralSimilarityIndexMeasure(range=1.0, reduction='none')

def train_model(config):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Create and configure the model using the provided hyperparameters
    batch_size = config["batch_size"]
    lr = config["lr"]
    model = SwinTransformer().to(device)
    
    # DataPrallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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
    mean_data = [201.863, 156.647, 188.265]
    std_data = [40.689, 52.119, 37.961]
    mean_label = [4.029, 10.271, 3.883, 11.188]
    std_label = [9.946, 15.307, 4.273, 14.874]

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_data, std=std_data)
    ])
    label_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_label, std=std_label)
    ])

    # Create instances of the ImageToImageDataset for the training, validation, and test sets
    train_path = r'/net/scratch2/t18155cb/data_summary/train'
    val_path = r'/net/scratch2/t18155cb/data_summary/val'

    train_dataset = ImageToImageDataset(train_path, input_transform=input_transform,
                                        label_transform=label_transform)
    val_dataset = ImageToImageDataset(val_path, input_transform=input_transform, label_transform=label_transform)


    # Create instances of the DataLoader for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    scaler = GradScaler()  # Initialize the GradScaler

    # Train your model using the optimizer and scheduler
    for epoch in range(10):  # Loop over the dataset multiple times
        print('Begin training')
        running_loss = 0.0
        epoch_ssim = []
        for i, (source, target) in enumerate(train_loader):
            inputs, labels = source.to(device), target.to(device)

            optimizer.zero_grad()
            
            # Wrap the forward pass in autocast context for mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Scale the loss and call backward() for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()

            running_loss += loss.item()

            scheduler.step()
    # Evaluate your model and report the performance using tune.report()
        # validation loop
        model.eval()
        val_loss = 0.0
        val_ssim = []
        with torch.no_grad():
            print('begin val')
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

        tune.report(loss=running_loss, accuracy=float(avg_val_ssim), iter_num=epoch)


def main(num_samples=50, max_num_epochs=10, gpus_per_trial=2):
    # Define the search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "lr_scheduler": tune.choice(["StepLR", "ExponentialLR", "CosineAnnealingLR"]),
        "batch_size": tune.choice([1, 2]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "ssim", "iteration"])
    result = tune.run(
        # tune.with_parameters(train, Model=net),
        partial(train_model),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    # Get the best hyperparameters
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    print("Best hyperparameters found were:", best_config)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    #args = get_args()
    #net = get_transNet(3)
    main(num_samples=20, max_num_epochs=10, gpus_per_trial=2)