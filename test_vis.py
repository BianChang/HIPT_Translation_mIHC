import tifffile
import torch
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import DataLoader
from SwinVisionTranformer import SwinTransformer, CustomSwinTransformer
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
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def test_model(model, test_loader, device, output_dir, label_dir):
    #dapi_t, cd3_t, cd20_t, panck_t= 0, 0, 0, 0
    #corr_coef_dapi_t, corr_coef_cd3_t, corr_coef_cd20_t, corr_coef_panck_t = 0, 0, 0, 0
    #psnr_dapi_t, psnr_cd3_t, psnr_cd20_t, psnr_panck_t = 0, 0, 0, 0
    dapi_t, cd3_t, panck_t= 0, 0, 0
    corr_coef_dapi_t, corr_coef_cd3_t, corr_coef_panck_t = 0, 0, 0
    psnr_dapi_t, psnr_cd3_t, psnr_panck_t = 0, 0, 0
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
            test_mask = test_mask.detach().cpu()
            predict1 = predict1.detach().cpu()
            ssim_4_channel_test = calculate_ssim_per_channel(predict1, test_mask)
            corr_coef_4_channel, psnr_4_channel = calculate_pearson_corr(predict1, test_mask)

            dapi_t += ssim_4_channel_test[2]
            cd3_t += ssim_4_channel_test[1]
            # cd20_t += ssim_4_channel_test[2]
            panck_t += ssim_4_channel_test[0]
            corr_coef_dapi_t += corr_coef_4_channel[2]
            corr_coef_cd3_t += corr_coef_4_channel[1]
            # corr_coef_cd20_t += corr_coef_4_channel[2]
            corr_coef_panck_t += corr_coef_4_channel[0]
            psnr_dapi_t += psnr_4_channel[2]
            psnr_cd3_t += psnr_4_channel[1]
            # psnr_cd20_t += psnr_4_channel[2]
            psnr_panck_t += psnr_4_channel[0]

            # Save the output tensors
            for output, filename in zip(predict1, filenames):
                output = output.squeeze()
                save_outputs(output, filename, output_dir)
            for label, filename in zip(test_mask, filenames):
                label = label.squeeze()
                save_outputs(label, filename, label_dir)


        dapi_t_mean = dapi_t / len(test_loader_num)
        cd3_t_mean = cd3_t / len(test_loader_num)
        # cd20_t_mean = cd20_t / len(test_loader_num)
        panck_t_mean = panck_t / len(test_loader_num)
        average_mean = (dapi_t_mean + cd3_t_mean + panck_t_mean) / 3
        corr_coef_dapi_t_mean = corr_coef_dapi_t / len(test_loader_num)
        corr_coef_cd3_t_mean = corr_coef_cd3_t / len(test_loader_num)
        # corr_coef_cd20_t_mean = corr_coef_cd20_t / len(test_loader_num)
        corr_coef_panck_t_mean = corr_coef_panck_t / len(test_loader_num)
        average_corr_coef_t_mean = (corr_coef_dapi_t_mean + corr_coef_cd3_t_mean + corr_coef_panck_t_mean) / 3
        psnr_dapi_t_mean = psnr_dapi_t / len(test_loader_num)
        psnr_cd3_t_mean = psnr_cd3_t / len(test_loader_num)
        # psnr_cd20_t_mean = psnr_cd20_t / len(test_loader_num)
        psnr_panck_t_mean = psnr_panck_t / len(test_loader_num)
        average_psnr_t_mean = (psnr_dapi_t_mean + psnr_cd3_t_mean + psnr_panck_t_mean) / 3

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
        logger.info('DAPI mean SSIM: {:.3f}'.format(dapi_t_mean))
        logger.info('CD3 mean SSIM: {:.3f}'.format(cd3_t_mean))
        # logger.info('CD20 mean SSIM: {:.3f}'.format(cd20_t_mean))
        logger.info('PanCK mean SSIM: {:.3f}'.format(panck_t_mean))
        logger.info('Average mean SSIM: {:.3f}'.format(average_mean))
        logger.info('-------')
        logger.info('DAPI mean PEARSON CORR: {:.3f}'.format(corr_coef_dapi_t_mean))
        logger.info('CD3 mean PEARSON CORR: {:.3f}'.format(corr_coef_cd3_t_mean))
        # logger.info('CD20 mean PEARSON CORR: {:.3f}'.format(corr_coef_cd20_t_mean))
        logger.info('PANCK mean PEARSON CORR: {:.3f}'.format(corr_coef_panck_t_mean))
        logger.info('Average mean PEARSON CORR: {:.3f}'.format(average_corr_coef_t_mean))
        logger.info('-------')
        logger.info('DAPI mean PSNR: {:.3f}'.format(psnr_dapi_t_mean))
        logger.info('CD3 mean PSNR: {:.3f}'.format(psnr_cd3_t_mean))
        # logger.info('CD20 mean PSNR: {:.3f}'.format(psnr_cd20_t_mean))
        logger.info('PANCK mean PSNR: {:.3f}'.format(psnr_panck_t_mean))
        logger.info('Average PSNR: {:.3f}'.format(average_psnr_t_mean))

        # Make sure to close the logger when you're done to free up resources
        logger.handlers.clear()

    return dapi_t_mean, cd3_t_mean, panck_t_mean, average_mean, \
        corr_coef_dapi_t_mean, corr_coef_cd3_t_mean, \
        corr_coef_panck_t_mean, average_corr_coef_t_mean,\
        psnr_dapi_t_mean, psnr_cd3_t_mean, psnr_panck_t_mean, average_psnr_t_mean


def save_outputs(output, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Remove extension of filename
    base_filename = os.path.splitext(filename)[0]

    # Convert tensor to numpy array and save as .tif
    output = output.permute(1, 2, 0)
    # Convert the image values back to [0, 1] from [-1, 1]
    output = (output + 1.0) / 2.0
    # Convert the image values back to [0, 255] from [0, 1]
    output = (output.cpu().detach().numpy() * 255).astype(np.uint8)

    '''
    # For 4 channel output
    # Define a dictionary of colors to be used for each channel
    colors = {0: [0, 0, 255], 1: [0, 255, 0], 2: [255, 255, 0], 3: [255, 0, 0]}

    channel_names = ['dapi', 'cd3', 'cd20', 'panck']
    for i, channel_name in enumerate(channel_names):
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        channel_path = os.path.join(channel_dir, f'{base_filename}.tif')

        # Create a blank (black) RGB image
        colored_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        # Assign the color to the target channel
        colored_output[:, :, :] = colors[i]
        # Multiply the color with the normalized intensity of the target channel
        colored_output = colored_output * (output[:, :, i, np.newaxis] / 255)
        colored_output = colored_output.astype(np.uint8)
        imwrite(channel_path, colored_output)

    multi_channel_dir = os.path.join(output_dir, 'multi_channel')
    os.makedirs(multi_channel_dir, exist_ok=True)
    output_path = os.path.join(multi_channel_dir, f'{base_filename}.tif')
    imwrite(output_path, output)
    '''
    colors = {0: [255, 0, 0], 1: [0, 255, 0], 3: [0, 0, 255]}

    channel_names = ['panck', 'cd3', 'dapi']
    for i, channel_name in enumerate(channel_names):
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        channel_path = os.path.join(channel_dir, f'{base_filename}.tif')

        # Create a blank (black) RGB image
        colored_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        # Assign the color to the target channel
        colored_output[:, :, :] = colors[i]
        # Multiply the color with the normalized intensity of the target channel
        colored_output = colored_output * (output[:, :, i, np.newaxis] / 255)
        colored_output = colored_output.astype(np.uint8)
        imwrite(channel_path, colored_output)

    multi_channel_dir = os.path.join(output_dir, 'multi_channel')
    os.makedirs(multi_channel_dir, exist_ok=True)
    output_path = os.path.join(multi_channel_dir, f'{base_filename}.tif')
    imwrite(output_path, output)



def main():
    # Define transforms
    '''
    mean_data = [x / 255. for x in [220.01547782, 191.56385728, 212.98354594]]
    std_data = [x / 255. for x in [40.00758663, 50.92426149, 35.41413304]]
    mean_label = [x / 255. for x in [0.10220867, 10.87440873, 1.94304308, 15.15272538]]
    std_label = [x / 255. for x in [1.4342306, 11.01720706, 4.51241098, 16.71110848]]
    '''

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
            "img_size": [1024, 1024],
            "patch_size": 32,
            "window_size": 64,
            "depths": [2, 2, 6, 2],
            "embed_dim": 96,
            "pretrained": False,
        }
    }
    model = SwinTransformer(**config["model_params"]).to(device)
    # model = CustomSwinTransformer(**config["model_params"]).to(device)

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
    dapi_t_mean, cd3_t_mean, panck_t_mean, average_mean, \
        corr_coef_dapi_mean, corr_coef_cd3_mean, corr_coef_panck_mean, average_pr, \
        psnr_dapi, psnr_cd3, psnr_panck, average_psnr= test_model(model, test_loader, device, output_path, label_path)
    # Print the results
    print('DAPI mean SSIM: {:.3f}'.format(dapi_t_mean))
    print('CD3 mean SSIM: {:.3f}'.format(cd3_t_mean))
    #print('CD20 mean SSIM: {:.3f}'.format(cd20_t_mean))
    print('PanCK mean SSIM: {:.3f}'.format(panck_t_mean))
    print('Average mean SSIM: {:.3f}'.format(average_mean))

    print('DAPI pearson corr: {:.3f}'.format(corr_coef_dapi_mean))
    print('CD3 pearson corr: {:.3f}'.format(corr_coef_cd3_mean))
    #print('CD20 pearson corr: {:.3f}'.format(corr_coef_cd20_mean))
    print('PANCK pearson corr: {:.3f}'.format(corr_coef_panck_mean))
    print('Average pearson corr: {:.3f}'.format(average_pr))

    print('DAPI psnr: {:.3f}'.format(psnr_dapi))
    print('CD3 psnr: {:.3f}'.format(psnr_cd3))
    #print('CD20 psnr: {:.3f}'.format(psnr_cd20))
    print('PANCK psnr: {:.3f}'.format(psnr_panck))
    print('AVERAGE psnr: {:.3f}'.format(average_psnr))

    # Visualize outputs
    visualized_output_path = os.path.join('./visualization', args.test_name, 'preds')
    os.makedirs(visualized_output_path, exist_ok=True)
    visualize_4channel_tif(os.path.join(output_path, 'multi_channel'), visualized_output_path)
    # Visualize labels
    visualized_labels_path = os.path.join('./visualization', args.test_name, 'labels')
    os.makedirs(visualized_labels_path, exist_ok=True)
    visualize_4channel_tif(os.path.join(label_path, 'multi_channel'),  visualized_labels_path)
    #output pdf for visualization
    # find common files in both directories
    files1 = set(os.listdir(visualized_output_path))
    files2 = set(os.listdir(visualized_labels_path))
    common_files = files1.intersection(files2)
    # create a pdf
    with PdfPages(os.path.join('./visualization', args.test_name,'output.pdf')) as pdf:
        for file in sorted(common_files):
            # open images
            img1 = Image.open(os.path.join(visualized_output_path, file))
            img2 = Image.open(os.path.join(visualized_labels_path, file))

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

