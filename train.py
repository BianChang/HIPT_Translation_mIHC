import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from SwinVisionTranformer import SwinTransformer, CustomSwinTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
from dataset.ImageToImageDataset import ImageToImageDataset
from utils.metrics import calculate_ssim_per_channel, calculate_pearson_corr
from tensorboardX import SummaryWriter
import argparse
import warnings
from time import time
from tqdm import tqdm


y_loss = {}  # loss history
y_ssim_dapi = {}
y_ssim_cd3 = {}
y_ssim_cd20 = {}
y_ssim_panck = {}
y_ssim_average = {}

y_loss['train'] = []
y_loss['val'] = []

y_ssim_dapi['train'] = []
y_ssim_cd3['train'] = []
y_ssim_cd20['train'] = []
y_ssim_panck['train'] = []
y_ssim_average['train'] = []

y_ssim_dapi['val'] = []
y_ssim_cd3['val'] = []
y_ssim_cd20['val'] = []
y_ssim_panck['val'] = []
y_ssim_average['val'] = []

y_ssim_dapi['test'] = []
y_ssim_cd3['test'] = []
y_ssim_cd20['test'] = []
y_ssim_panck['test'] = []
y_ssim_average['test'] = []

x_epoch = []
fig = plt.figure()


def train(net=None):
    pre_model = args.premodel
    bs_p_card = args.batchsize
    lr = args.lr
    epoch_num = args.epoch
    model_name = args.name

    warnings.filterwarnings("ignore")

    BATCHSIZE_PER_CARD = int(bs_p_card)

    if pre_model.endswith('.th'):
        net.load_state_dict(torch.load(pre_model))
    else:
        pass

    batch_size = BATCHSIZE_PER_CARD

    # Define the transforms for the input and label images
    '''
    mean_data = [x / 255. for x in [220.01547782, 191.56385728, 212.98354594]]
    std_data = [x / 255. for x in [40.00758663, 50.92426149, 35.41413304]]
    mean_label = [x / 255. for x in [0.10220867, 10.87440873, 1.94304308, 15.15272538]]
    std_label = [x / 255. for x in [1.4342306, 11.01720706, 4.51241098, 16.71110848]]
    '''

    mean_data = [0.5, 0.5, 0.5]
    std_data = [0.5, 0.5, 0.5]
    mean_label = [0.5, 0.5, 0.5, 0.5]
    std_label = [0.5, 0.5, 0.5, 0.5]

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_data, std=std_data)
    ])
    label_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_label, std=std_label)
    ])

    # Create instances of the ImageToImageDataset for the training, validation, and test sets
    train_dataset = ImageToImageDataset(args.train_path, input_transform=input_transform, label_transform=label_transform)
    val_dataset = ImageToImageDataset(args.val_path, input_transform=input_transform, label_transform=label_transform)
    test_dataset = ImageToImageDataset(args.test_path, input_transform=input_transform, label_transform=label_transform)

    # Create instances of the DataLoader for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)

    # Visualize settings
    print('''Starting training:
           Epochs:          %.2f
           Batch size:      %.2f
           Learning rate:   %.5f_transform
           Training size:   %.0f
           Validation size: %.0f
           Test size: %.0f
       ''' % (int(epoch_num), int(bs_p_card), float(lr), n_train, n_val, n_test))

    writer = SummaryWriter('./record')
    # Check if the directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    mylog = open('logs/' + model_name + '.log', 'w')
    tic = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_optim = 0
    total_epoch = int(epoch_num)
    train_epoch_best_loss = 100.
    val_epoch_best_loss = 100.
    best_batch_ssim_val = 0
    criterion = torch.nn.L1Loss()

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.to(device)
    else:
        net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)


    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:' + str(epoch) + ' ----------')
        data_loader_iter = train_loader
        train_epoch_loss = 0.
        dapi_train, cd3_train, cd20_train, panck_train, average_train = 0, 0, 0, 0, 0
        print('Train:')
        for img, mask in tqdm(data_loader_iter, ncols=20, total=len(data_loader_iter)):

            net.train()
            lr = scheduler.get_lr()
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = net(img)
            train_loss = criterion(pred, mask)

            pred_cpu = pred.detach().cpu()
            mask_cpu = mask.detach().cpu()

            ssim_4_channel_train = calculate_ssim_per_channel(pred_cpu, mask_cpu)
            dapi_train += ssim_4_channel_train[0]
            cd3_train += ssim_4_channel_train[1]
            cd20_train += ssim_4_channel_train[2]
            panck_train += ssim_4_channel_train[3]
            average_train += np.mean(ssim_4_channel_train)
            train_epoch_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        scheduler.step()
        train_epoch_loss /= len(data_loader_iter)

        val_loader_num = val_loader
        test_loader_num = test_loader

        val_epoch_loss = 0

        dapi_v, cd3_v, cd20_v, panck_v, average_v = 0, 0, 0, 0, 0
        dapi_t, cd3_t, cd20_t, panck_t, average_t = 0, 0, 0, 0, 0

        # Validation
        print('Validation:')
        with torch.no_grad():
            for val_img, val_mask in tqdm(val_loader_num, ncols=20, total=len(val_loader_num)):
                val_img, val_mask = val_img.to(device), val_mask.to(device)
                net.eval()
                predict = net(val_img)
                val_loss = criterion(predict, val_mask)

                predict = predict.detach().cpu()
                val_mask = val_mask.detach().cpu()

                val_epoch_loss += val_loss
                ssim_4_channel_val = calculate_ssim_per_channel(predict,val_mask)
                dapi_v += ssim_4_channel_val[0]
                cd3_v += ssim_4_channel_val[1]
                cd20_v += ssim_4_channel_val[2]
                panck_v += ssim_4_channel_val[3]
                average_v += np.mean(ssim_4_channel_val)

            # Test
            print('Test:')
            for test_img, test_mask in tqdm(test_loader_num, ncols=20, total=len(test_loader_num)):
                test_img, test_mask = test_img.to(device), test_mask.to(device)
                net.eval()
                predict1 = net(test_img)
                test_mask = test_mask.detach().cpu()
                predict1 = predict1.detach().cpu()

                ssim_4_channel_test = calculate_ssim_per_channel(predict1, test_mask)
                dapi_t += ssim_4_channel_test[0]
                cd3_t += ssim_4_channel_test[1]
                cd20_t += ssim_4_channel_test[2]
                panck_t += ssim_4_channel_test[3]
                average_t += np.mean(ssim_4_channel_test)

        batch_ssim_dapi_train = dapi_train / len(data_loader_iter)
        batch_ssim_cd3_train = cd3_train / len(data_loader_iter)
        batch_ssim_cd20_train = cd20_train / len(data_loader_iter)
        batch_ssim_panck_train = panck_train / len(data_loader_iter)
        batch_ssim_average_train = average_train / len(data_loader_iter)

        val_epoch_loss = val_epoch_loss / len(val_loader_num)
        val_epoch_loss = val_epoch_loss.detach().cpu().numpy()
        batch_ssim_dapi_val = dapi_v / len(val_loader_num)
        batch_ssim_cd3_val = cd3_v / len(val_loader_num)
        batch_ssim_cd20_val = cd20_v / len(val_loader_num)
        batch_ssim_panck_val = panck_v / len(val_loader_num)
        batch_ssim_average_val = average_v / len(val_loader_num)

        batch_ssim_dapi_test = dapi_t / len(test_loader_num)
        batch_ssim_cd3_test = cd3_t / len(test_loader_num)
        batch_ssim_cd20_test = cd20_t / len(test_loader_num)
        batch_ssim_panck_test = panck_t / len(test_loader_num)
        batch_ssim_average_test = average_t / len(test_loader_num)

        y_loss['train'].append(train_epoch_loss)
        y_loss['val'].append(val_epoch_loss)

        y_ssim_dapi['train'].append(batch_ssim_dapi_train)
        y_ssim_cd3['train'].append(batch_ssim_cd3_train)
        y_ssim_cd20['train'].append(batch_ssim_cd20_train)
        y_ssim_panck['train'].append(batch_ssim_panck_train)
        y_ssim_average['train'].append(batch_ssim_average_train)

        y_ssim_dapi['val'].append(batch_ssim_dapi_val)
        y_ssim_cd3['val'].append(batch_ssim_cd3_val)
        y_ssim_cd20['val'].append(batch_ssim_cd20_val)
        y_ssim_panck['val'].append(batch_ssim_panck_val)
        y_ssim_average['val'].append(batch_ssim_average_val)

        y_ssim_dapi['test'].append(batch_ssim_dapi_test)
        y_ssim_cd3['test'].append(batch_ssim_cd3_test)
        y_ssim_cd20['test'].append(batch_ssim_cd20_test)
        y_ssim_panck['test'].append(batch_ssim_panck_test)
        y_ssim_average['test'].append(batch_ssim_average_test)
        draw_curve(model_name, epoch)
        draw_ssim_curve_train(model_name, epoch)
        draw_ssim_curve_val(model_name, epoch)
        draw_ssim_curve_test(model_name, epoch)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train_loss', train_epoch_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

        writer.add_scalar('ssim_dapi_val', batch_ssim_dapi_val, epoch)
        writer.add_scalar('ssim_cd3_val', batch_ssim_cd3_val, epoch)
        writer.add_scalar('ssim_cd20_val', batch_ssim_cd20_val, epoch)
        writer.add_scalar('ssim_panck_val', batch_ssim_panck_val, epoch)
        writer.add_scalar('ssim_average_val', batch_ssim_average_val, epoch)
        # mylog.write('********** ' + 'lr={:.10f}'.format(scheduler.get_lr()[0]) + ' **********' + '\n')
        mylog.write('********** ' + 'lr={:.10f}'.format(optimizer.param_groups[0]['lr']) + ' **********' + '\n')
        mylog.write(
            '--epoch:' + str(epoch) + '  --time:' + str(int(time() - tic)) + '  --train_loss:' + str(train_epoch_loss)
            + ' --val_loss:' + str(val_loss) + '\n'
            + '--ssim_dapi:' + str(batch_ssim_dapi_val) + '--ssim_cd3:' + str(batch_ssim_cd3_val)
            + '--ssim_cd20:' + str(batch_ssim_cd20_val) + '--ssim_panck:' + str(
                batch_ssim_panck_val) + '--ssim_average:' + str(batch_ssim_average_val) + '\n'
            + '--ssim_dapi_test:' + str(batch_ssim_dapi_test) + '--ssim_cd3_test:' + str(batch_ssim_cd3_test)
            + '--ssim_cd20_test:' + str(batch_ssim_cd20_test) + '--ssim_panck_test:' + str(batch_ssim_panck_test)
            + '--ssim_average_test:' + str(batch_ssim_average_test) + '\n')
        print(
            '--epoch: {} --time: {} --train_loss: {:.3f} --val_loss: {:.3f} --ssim_average_val: {:.3f} --ssim_average_test: {:.3f}'.format(
                epoch, int(time() - tic), train_epoch_loss, val_epoch_loss, batch_ssim_average_val,
                batch_ssim_average_test))

        if not os.path.exists('./weights/' + model_name + '/'):
            os.mkdir('./weights/' + model_name + '/')
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss

            torch.save(net.state_dict(), 'weights/' + model_name + '/' + model_name + '_train_loss_best.th')
        if val_loss <= val_epoch_best_loss:
            val_epoch_best_loss = val_loss
            torch.save(net.state_dict(), 'weights/' + model_name + '/' + model_name + '_val_loss_best.th')
        if batch_ssim_average_val >= best_batch_ssim_val:
            best_batch_ssim_val = batch_ssim_average_val
            torch.save(net.state_dict(), 'weights/' + model_name + '/' + model_name + 'val_ssim_best.th')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'weights/' + model_name + '/' + model_name + '%d.th' % (epoch))

        mylog.flush()

    # writer.add_graph(Model(), img)
    print('Train Finish !')
    mylog.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the network on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--model_name', dest='name', type=str, default='default')
    parser.add_argument('-b', '--batch_size', dest='batchsize', type=int, default=4, help='batch size')
    parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=100, help='data root')
    parser.add_argument('-p', '--pretrained', dest='premodel', type=str, default='None', help='pre trained model')
    parser.add_argument('-tp', '--train_path', dest='train_path', type=str,
                        default=r'F:\2023_4_11_data_organization\Patches\small_dataset\train')
    parser.add_argument('-vp', '--val_path', dest='val_path', type=str,
                        default=r'F:\2023_4_11_data_organization\Patches\small_dataset\val')
    parser.add_argument('-test_p', '--test_path', dest='test_path', type=str,
                        default=r'F:\2023_4_11_data_organization\Patches\small_dataset\test')


    return parser.parse_args()

def draw_curve(model_name, current_epoch):
    x_epoch.append(current_epoch)
    fig1 = plt.figure("loss")
    plt.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    plt.plot(x_epoch, y_loss['val'], 'ro-', label='val')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    if current_epoch == 1:
        plt.legend()
    fig1.savefig(os.path.join('./lossGraphs', model_name + '_train.jpg'))

def draw_ssim_curve_val(model_name, current_epoch):
    fig2 = plt.figure("SSIM_val")
    plt.plot(x_epoch, y_ssim_dapi['val'], 'ro-', label='ssim_dapi_val')
    plt.plot(x_epoch, y_ssim_cd3['val'], 'go-', label='ssim_cd3_val')
    plt.plot(x_epoch, y_ssim_cd20['val'], 'bo-', label='ssim_cd20_val')
    plt.plot(x_epoch, y_ssim_panck['val'], 'co-', label='ssim_panck_val')
    plt.plot(x_epoch, y_ssim_average['val'], 'mo-', label='ssim_average_val')

    plt.xlabel('Epoch')
    plt.ylabel('SSIM_Validation')
    plt.title('Validation SSIM Curve')

    if current_epoch == 1:
        plt.legend()
    fig2.savefig(os.path.join('./lossGraphs', model_name + '_ssim_val.jpg'))

def draw_ssim_curve_test(model_name, current_epoch):
    fig3 = plt.figure("SSIM_test")
    plt.plot(x_epoch, y_ssim_dapi['test'], 'ro-', label='ssim_dapi_test')
    plt.plot(x_epoch, y_ssim_cd3['test'], 'go-', label='ssim_cd3_test')
    plt.plot(x_epoch, y_ssim_cd20['test'], 'bo-', label='ssim_cd20_test')
    plt.plot(x_epoch, y_ssim_panck['test'], 'co-', label='ssim_panck_test')
    plt.plot(x_epoch, y_ssim_average['test'], 'mo-', label='ssim_average_test')

    plt.xlabel('Epoch')
    plt.ylabel('SSIM_Test')
    plt.title('Test SSIM Curve')

    if current_epoch == 1:
        plt.legend()
    fig3.savefig(os.path.join('./lossGraphs', model_name + '_ssim_test.jpg'))

def draw_ssim_curve_train(model_name, current_epoch):
    fig4 = plt.figure("SSIM_train")
    plt.plot(x_epoch, y_ssim_dapi['train'], 'ro-', label='ssim_dapi_train')
    plt.plot(x_epoch, y_ssim_cd3['train'], 'go-', label='ssim_cd3_train')
    plt.plot(x_epoch, y_ssim_cd20['train'], 'bo-', label='ssim_cd20_train')
    plt.plot(x_epoch, y_ssim_panck['train'], 'co-', label='ssim_panck_train')
    plt.plot(x_epoch, y_ssim_average['train'], 'mo-', label='ssim_average_train')
    if current_epoch == 1:
        plt.legend()
    fig4.savefig(os.path.join('./lossGraphs', model_name + '_ssim_train.jpg'))

if __name__ == '__main__':
    args = get_args()
    config = {
        "model_params": {
            "img_size": [1024, 1024],
            "patch_size": 4,
            "window_size": 8,
            "depths": [2, 2, 6, 2],
            "embed_dim": 96,
            "pretrained": False,
        }
    }
    net = SwinTransformer(**config["model_params"])
    # net = CustomSwinTransformer(**config["model_params"])
    train(net)


