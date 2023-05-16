import tifffile
import torch
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import DataLoader
from SwinVisionTranformer import SwinTransformer
from dataset.ImageToImageDataset import ImageToImageDatasetWithName
from utils.visulization import visualize_4channel_tif
from utils.metrics import calculate_ssim_per_channel, calculate_pearson_corr
from PIL import Image
import os
import numpy as np
import tqdm
from tifffile import imsave
import logging
import datetime


def test_model(model, test_loader, device, output_dir):
    dapi_t, cd3_t, cd20_t, panck_t, average_t = 0, 0, 0, 0, 0
    model.eval()
    outputs = []
    test_loader_num = test_loader
    with torch.no_grad():
        print('Test:')
        for test_img, test_mask, filenames in tqdm(test_loader_num, ncols=20, total=len(test_loader_num)):
            test_img, test_mask = test_img.to(device), test_mask.to(device)
            model.eval()
            predict1 = model(test_img)
            test_mask = test_mask.detach().cpu()
            predict1 = predict1.detach().cpu()
            ssim_4_channel_test = calculate_ssim_per_channel(predict1, test_mask)

            dapi_t += ssim_4_channel_test[0]
            cd3_t += ssim_4_channel_test[1]
            cd20_t += ssim_4_channel_test[2]
            panck_t += ssim_4_channel_test[3]

            # Save the output tensors
            for output, filename in zip(predict1, filenames):
                save_outputs(output, filename, output_dir)

        dapi_t_mean = dapi_t / len(test_loader_num)
        cd3_t_mean = cd3_t / len(test_loader_num)
        cd20_t_mean = cd20_t / len(test_loader_num)
        panck_t_mean = panck_t / len(test_loader_num)
        average_mean = (dapi_t_mean + cd3_t_mean + cd20_t_mean + panck_t_mean) / 4

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string
        # This will give a string like "2023-05-16_12-30-00"
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Use the date and time string in your log filename
        log_filename = f"test_results_{datetime_str}.log"

        # Create a logging object
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Log the results
        logger.info('DAPI mean SSIM: %s', dapi_t_mean)
        logger.info('CD3 mean SSIM: %s', cd3_t_mean)
        logger.info('CD20 mean SSIM: %s', cd20_t_mean)
        logger.info('PanCK mean SSIM: %s', panck_t_mean)
        logger.info('Average mean SSIM: %s', average_mean)

        # Make sure to close the logger when you're done to free up resources
        logger.handlers.clear()

    return dapi_t_mean, cd3_t_mean, cd20_t_mean, panck_t_mean, average_mean


def save_outputs(output, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Remove extension of filename
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f'{base_filename}.tif')
    # Convert tensor to numpy array and save as .tif
    output = (output * 255).byte()
    tifffile.imwrite(output_path, output.numpy())


def main():
    # Define transforms
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

    # Define dataset and dataloader
    test_dataset = ImageToImageDatasetWithName(args.test_path, input_transform=input_transform, label_transform=label_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Define device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Test model
    dapi_t_mean, cd3_t_mean, cd20_t_mean, panck_t_mean, average_mean = test_model(model, test_loader, device)
    # Print the results
    print('DAPI mean SSIM:', dapi_t_mean)
    print('CD3 mean SSIM:', cd3_t_mean)
    print('CD20 mean SSIM:', cd20_t_mean)
    print('PanCK mean SSIM:', panck_t_mean)
    print('Average mean SSIM:', average_mean)

    # Visualize outputs
    visualize_4channel_tif(args.output_path, args.visualized_output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, help='Path to test data')
    parser.add_argument('--model_path', type=str, help='Path to the pretrained model')
    parser.add_argument('--output_path', type=str, help='Path to save the outputs')
    parser.add_argument('--visualized_output_path', type=str, help='Path to save the visualized outputs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')

    args = parser.parse_args()
    main()

