import tifffile
import torch
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import DataLoader
from SwinVisionTranformer import SwinTransformer, CustomSwinTransformer, HybridSwinT
from dataset.ImageToImageDataset import ImageToImageDatasetWithName
from utils.visulization import visualize_4channel_tif
from utils.metrics import calculate_ssim_per_channel, calculate_pearson_corr
from PIL import Image
from tifffile import imread, imwrite
import os
import numpy as np
from tqdm import tqdm
from tifffile import imsave
import logging
import datetime
import torch.nn
import cv2 as cv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def test_model(model, test_loader, device, output_dir, label_dir):

    model.eval()
    test_loader_num = test_loader
    with torch.no_grad():
        print('Test:')
        for test_img, test_mask, filenames in tqdm(test_loader_num, ncols=20, total=len(test_loader_num)):
            test_img, test_mask = test_img.to(device), test_mask.to(device)
            model.eval()
            predict1 = model(test_img)
            test_mask = test_mask.detach().cpu()
            predict1 = predict1.detach().cpu()

            # Save the output tensors
            for output, filename in zip(predict1, filenames):
                output = output.squeeze()
                save_outputs(output, filename, output_dir)
            for label, filename in zip(test_mask, filenames):
                label = label.squeeze()
                save_outputs(label, filename, label_dir)

        average_psnr, average_ssim, average_pearson = psnr_and_ssim_and_pearson(output_dir, label_dir)
        # Define the directory where you want to save the log files
        log_dir = "./test_logs"

        # Make sure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string
        # This will give a string like "2023-05-16_12-30-00"
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Use the date and time string in your log filename
        # log_filename = f"test_results_{datetime_str}.log"
        log_filename = f"{args.test_name}_{datetime_str}.log"

        # Combine the directory with the filename
        log_filepath = os.path.join(log_dir, log_filename)

        # Create a logging object
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler(log_filepath)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Log the results
        logger.info('average_psnr: {:.3f}'.format(average_psnr))
        logger.info('average_ssim: {:.3f}'.format(average_ssim))
        logger.info('average_pearson: {:.3f}'.format(average_pearson))
        # Make sure to close the logger when you're done to free up resources
        logger.handlers.clear()


def save_outputs(output, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Remove extension of filename
    base_filename = os.path.splitext(filename)[0]

    # Convert tensor to numpy array and save as .png
    output = output.permute(1, 2, 0)
    # Convert the image values back to [0, 1] from [-1, 1]
    output = (output + 1.0) / 2.0
    # Convert the image values back to [0, 255] from [0, 1]
    output = (output.cpu().detach().numpy() * 255).astype(np.uint8)

    # Convert the image to RGB format
    output_rgb = cv.cvtColor(output, cv.COLOR_BGR2RGB)

    output_path = os.path.join(output_dir, f'{base_filename}.png')
    cv.imwrite(output_path, output_rgb)


def psnr_and_ssim_and_pearson(output_dir, label_dir):
    psnr = []
    ssim = []
    pearson_correlations = []

    for i in tqdm(os.listdir(os.path.join(output_dir))):
        if '.png' in i:
            try:
                fake = cv.imread(os.path.join(output_dir, i))
                real = cv.imread(os.path.join(label_dir, i))
                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)
                SSIM = structural_similarity(fake, real, multichannel=True)
                ssim.append(SSIM)

                # Calculate Pearson correlation directly on flattened images
                fake_flat = fake.flatten()
                real_flat = real.flatten()
                corr_coef_matrix = np.corrcoef(fake_flat, real_flat)
                pearson_c = corr_coef_matrix[0, 1]  # correlation coefficient
                pearson_correlations.append(pearson_c)

            except:
                print("There is something wrong with " + i)
        else:
            continue

    average_psnr = sum(psnr) / len(psnr)
    average_ssim = sum(ssim) / len(ssim)
    average_pearson = sum(pearson_correlations) / len(pearson_correlations)

    print("The average PSNR is " + str(average_psnr))
    print("The average SSIM is " + str(average_ssim))
    print("The average Pearson correlation is " + str(average_pearson))

    return average_psnr, average_ssim, average_pearson


def main():
    # Define transforms
    mean_data = [0.5, 0.5, 0.5]
    std_data = [0.5, 0.5, 0.5]
    mean_label = [0.5, 0.5, 0.5]
    std_label = [0.5, 0.5, 0.5]

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
    config = {
        "model_params": {
            "img_size": [256, 256],
            "patch_size": 2,
            "window_size": 4,
            "depths": [2, 2, 6, 2],
            "embed_dim": 96,
            "pretrained": False,
        }
    }
    # model = SwinTransformer(**config["model_params"]).to(device)
    # model = CustomSwinTransformer(**config["model_params"]).to(device)
    model = HybridSwinT(**config["model_params"]).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    # If the model is an instance of nn.DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model_path = os.path.join('./output', args.test_name)
    os.makedirs(model_path, exist_ok=True)

    output_path = os.path.join('./output', args.test_name, 'preds')
    os.makedirs(output_path, exist_ok=True)

    label_path = os.path.join('./output', args.test_name, 'labels')
    os.makedirs(output_path, exist_ok=True)

    # Test model
    test_model(model, test_loader, device, output_path, label_path)


    files1 = set(os.listdir(os.path.join('./output', args.test_name, 'preds')))
    files2 = set(os.listdir(os.path.join('./output', args.test_name, 'labels')))
    common_files = files1.intersection(files2)
    # create a pdf
    os.makedirs(os.path.join('./visualization', args.test_name), exist_ok=True)
    with PdfPages(os.path.join('./visualization', args.test_name,'output.pdf')) as pdf:
        for file in sorted(common_files):
            # open images
            img1 = Image.open(os.path.join(os.path.join('./output', args.test_name, 'preds'), file))
            img2 = Image.open(os.path.join(os.path.join('./output', args.test_name, 'labels'), file))

            # create a figure and show images
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img1)
            axs[0].set_title('Output: ' + file)
            axs[0].axis('off')
            axs[1].imshow(img2)
            axs[1].set_title('Label: ' + file)
            axs[1].axis('off')

            # save the figure to pdf
            pdf.savefig(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help='Name of the test trial')
    parser.add_argument('--test_path', type=str, help='Path to test data')
    parser.add_argument('--model_path', type=str, help='Path to the pretrained model')
    parser.add_argument('--output_path', type=str, help='Path to save the outputs')
    parser.add_argument('--visualized_output_path', type=str, help='Path to save the visualized outputs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')

    args = parser.parse_args()
    main()

