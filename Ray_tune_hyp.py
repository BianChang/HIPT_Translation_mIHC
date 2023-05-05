import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from SwinVisionTranformer import SwinTransformer, Decoder
import numpy as np
from dataset.ImageToImageDataset import ImageToImageDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from utils.metrics import calculate_ssim_per_channel


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # Create and configure the model using the provided hyperparameters
    batch_size = config["batch_size"]
    lr = config["lr"]

    model = SwinTransformer(**config["model_params"])
    
    # DataPrallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if config["lr_scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif config["lr_scheduler"] == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif config["lr_scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = nn.L1Loss()

    # Load and preprocess the dataset using the provided patch_size
    # Define the transforms for the input and label images
    '''
    mean_data = [201.863, 156.647, 188.265]
    std_data = [40.689, 52.119, 37.961]
    mean_label = [4.029, 10.271, 3.883, 11.188]
    std_label = [9.946, 15.307, 4.273, 14.874]
    '''
    '''
    # 512 patches
    mean_data = [x / 255. for x in [207.829, 166.691, 195.421]]
    std_data = [x / 255. for x in [40.153, 51.167, 36.759]]
    mean_label = [x / 255. for x in [1.944, 10.204, 3.079, 10.236]]
    std_label = [x / 255. for x in [6.419, 14.140, 4.142, 14.107]]
    '''

    #224 patches
    mean_data = [x / 255. for x in [220.01547782, 191.56385728, 212.98354594]]
    std_data = [x / 255. for x in [40.00758663, 50.92426149, 35.41413304]]
    mean_label = [x / 255. for x in [0.10220867, 10.87440873, 1.94304308, 15.15272538]]
    std_label = [x / 255. for x in [1.4342306, 11.01720706, 4.51241098, 16.71110848]]



    input_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_data, std=std_data)
    ])
    label_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_label, std=std_label)
    ])

    # Create instances of the ImageToImageDataset for the training, validation, and test sets
    train_path = r'F:\2023_4_11_data_organization\224_patches\merged\train'
    val_path = r'F:\2023_4_11_data_organization\224_patches\merged\val'

    train_dataset = ImageToImageDataset(train_path, input_transform=input_transform,
                                        label_transform=label_transform)
    val_dataset = ImageToImageDataset(val_path, input_transform=input_transform, label_transform=label_transform)


    # Create instances of the DataLoader for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    scaler = GradScaler()  # Initialize the GradScaler

    # Train your model using the optimizer and scheduler
    for epoch in range(5):  # Loop over the dataset multiple times
        running_loss = 0.0
        epoch_ssim = []
        for i, (source, target) in enumerate(train_loader):
            inputs, labels = source.to(device), target.to(device)

            optimizer.zero_grad()
            
            # Wrap the forward pass in autocast context for mixed precision
            '''  
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

            
            # Scale the loss and call backward() for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
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
            print('begin val')
            for i, (source, target) in enumerate(val_loader):
                source, target = source.to(device), target.to(device)

                output = model(source)
                loss = criterion(output, target)
                val_loss += loss.item()

                # Calculate SSIM
                ssim_val = calculate_ssim_per_channel(output, target)
                #print(ssim_val)
                val_ssim.append(ssim_val)
        running_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_ssim = np.mean(val_ssim, axis=0)
        avg_val_ssim = np.mean(val_ssim)

    tune.report(train_loss=running_loss, val_loss=val_loss, val_ssim=float(avg_val_ssim), iter_num=epoch)


def main(num_samples=50, max_num_epochs=10, gpus_per_trial=1):
    # Define the search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "lr_scheduler": tune.choice(["StepLR", "ExponentialLR", "CosineAnnealingLR"]),
        "batch_size": tune.choice([1, 2]),
        "model_params": {
            "img_size": [224, 224],
            "patch_size": 4,
            "window_size": 7,
        }
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    search_alg = AxSearch(metric='loss', mode='min')
    # Wrap your search algorithm with ConcurrencyLimiter
    limited_search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)  # Set max_concurrent to your desired value

    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss", "val_ssim", "iter_num"])
    result = tune.run(
        # tune.with_parameters(train, Model=net),
        partial(train_model),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        search_alg=limited_search_alg,
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
    main(num_samples=20, max_num_epochs=5, gpus_per_trial=1)