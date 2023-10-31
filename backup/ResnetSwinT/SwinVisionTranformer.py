import torch
import torch.nn as nn
from torch import einsum
from timm.models.layers import DropPath, to_2tuple
from timm.models.swin_transformer import SwinTransformerBlock
import math
from torch.utils.checkpoint import checkpoint
from timm.models.swin_transformer import PatchMerging, PatchEmbed
import timm
import functools
from torch.nn import init
from einops import rearrange, repeat
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channs, output_channels, patch_size):
        super().__init__()

        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        upsampling_factor = int(math.log(patch_size, 2))
        self.final_upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(in_channs // 8, output_channels if i == upsampling_factor - 1 else in_channs // 8,
                                kernel_size=2, stride=2) for i
             in range(upsampling_factor)])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, stage_outputs):
        x = self.upsample1(x)
        h_w_dim = int((stage_outputs[-1].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-1].view(stage_outputs[-1].shape[0], h_w_dim, h_w_dim, stage_outputs[-1].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv1(x)

        x = self.upsample2(x)
        h_w_dim2 = int((stage_outputs[-2].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-2].view(stage_outputs[-2].shape[0], h_w_dim2, h_w_dim2, stage_outputs[-2].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv2(x)

        x = self.upsample3(x)
        h_w_dim3 = int((stage_outputs[-3].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-3].view(stage_outputs[-3].shape[0], h_w_dim3, h_w_dim3, stage_outputs[-3].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv3(x)

        for upsample_layer in self.final_upsample_layers:
            x = upsample_layer(x)

        return self.tanh(x)


class Decoder_hbrid(nn.Module):
    def __init__(self, in_channs, output_channels, patch_size):
        super().__init__()

        '''
        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)
        '''

        # Replace ConvTranspose with Upsample + Convolution
        self.upsample_and_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)
        )
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample_and_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs // 2, in_channs // 4, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs // 4, kernel_size=3, padding=1)

        self.upsample_and_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        #upsampling_factor = int(math.log(patch_size, 2)) + 3
        '''
        self.final_upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(in_channs // 8, output_channels if i == upsampling_factor - 1 else in_channs // 8,
                                kernel_size=2, stride=2) for i
             in range(upsampling_factor)])
        '''


        self.upsample_and_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1)
        )
        self.upsample_and_conv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1)
        )

        self.conv4 = nn.Conv2d(88, 44, kernel_size=3, padding=1)

        self.upsample_and_conv6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(44, 22, kernel_size=3, padding=1)
        )

        self.conv5 = nn.Conv2d(54, 27, kernel_size=3, padding=1)

        self.upsample_and_conv7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(27, 14, kernel_size=3, padding=1)
        )

        self.conv6 = nn.Conv2d(30, 15, kernel_size=3, padding=1)

        self.upsample_and_conv8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(15, 3, kernel_size=3, padding=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, stage_outputs):
        x = self.upsample_and_conv1(x)
        h_w_dim = int((stage_outputs[-1].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-1].view(stage_outputs[-1].shape[0], h_w_dim, h_w_dim, stage_outputs[-1].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv1(x)

        x = self.upsample_and_conv2(x)
        h_w_dim2 = int((stage_outputs[-2].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-2].view(stage_outputs[-2].shape[0], h_w_dim2, h_w_dim2, stage_outputs[-2].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv2(x)

        x = self.upsample_and_conv3(x)
        h_w_dim3 = int((stage_outputs[-3].shape[1]) ** 0.5)
        x = torch.cat((x, stage_outputs[-3].view(stage_outputs[-3].shape[0], h_w_dim3, h_w_dim3, stage_outputs[-3].shape[2]).permute(0, 3, 1, 2)), dim=1)
        x = self.conv3(x)

        x = self.upsample_and_conv4(x)
        x = self.upsample_and_conv5(x)

        x = torch.cat((x, stage_outputs[-4]), dim=1)
        x = self.conv4(x)

        x = self.upsample_and_conv6(x)

        x = torch.cat((x, stage_outputs[-5]), dim=1)
        x = self.conv5(x)

        x = self.upsample_and_conv7(x)

        x = torch.cat((x, stage_outputs[-6]), dim=1)
        x = self.conv6(x)

        x = self.upsample_and_conv8(x)


        return self.tanh(x)


'''
class SwinTransformer(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=3, window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=4, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        input_resolutions = [(img_size[0] // patch_size, img_size[1] // patch_size)]
        for _ in range(1, len(depths)):
            input_resolutions.append((input_resolutions[-1][0] // 2, input_resolutions[-1][1] // 2))

        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, embed_dim))
        # nn.init.kaiming_uniform_(self.pos_embed, a=math.sqrt(5))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks_and_merging = nn.ModuleList([])

        for i in range(len(depths)):
            stage_blocks = [
                SwinTransformerBlock(
                    dim=embed_dim * 2 ** i, input_resolution=input_resolutions[i],
                    num_heads=num_heads, window_size=window_size, shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate, norm_layer=norm_layer)
                for _ in range(depths[i])
            ]

            self.blocks_and_merging.append(nn.Sequential(*stage_blocks))

            if i < len(depths) - 1:  # Don't add patch merging after the last stage
                patch_merging = PatchMerging(input_resolution=input_resolutions[i],
                                             dim=embed_dim * 2 ** i, norm_layer=norm_layer)
                self.blocks_and_merging.append(patch_merging)

        self.decoder = Decoder(in_channs=int(embed_dim * (math.pow(2, len(depths)-1))), output_channels=output_channels,
                               )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.stage_outputs = []
        # Apply patch embedding to convert the input image into a sequence of flattened patches
        # print("input shape:", x.shape)
        x = self.patch_embed(x)
        # Extract the batch size (B)
        B, N, W = x.shape
        # Add positional encoding to the patch embeddings
        x = x + self.pos_embed
        # Apply dropout to the patch embeddings (prevent overfitting)
        x = self.pos_drop(x)

        # Process the patch embeddings through the Swin Transformer blocks
        for i, layer in enumerate(self.blocks_and_merging):
            x = layer(x)
            #print(i, x.shape)
            if i + 1 < len(self.blocks_and_merging) and isinstance(self.blocks_and_merging[i + 1], PatchMerging):
                # Save the output of each stage before the patch merging in self.stage_outputs
                self.stage_outputs.append(x)

        # Reshape the output tensor to obtain the image features in the original spatial dimensions
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths)-1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths)-1)), self.last_stage_dim)
        # Permute the tensor dimensions to make it compatible with the decoder
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)

        x = self.decoder(x, self.stage_outputs)

        return x
'''


class SwinTransformer(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=3, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))

        # create the Swin Transformer model
        self.model = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.model.head = nn.Identity()  # remove classification head

        self.decoder = Decoder(in_channs=int(embed_dim * (math.pow(2, len(depths) - 1))),
                               output_channels=output_channels,
                               patch_size=patch_size,
                               )

    def forward(self, x):
        self.stage_outputs = []
        B, C, H, W = x.shape
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for stage in self.model.layers:
            for blk in stage.blocks:
                x = blk(x)
            if stage.downsample is not None:
                self.stage_outputs.append(x)
                x = stage.downsample(x)

        x = self.model.norm(x)
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths) - 1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths) - 1)), self.last_stage_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x, self.stage_outputs)

        return x


class CustomSwinTransformer(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=3, window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=4, pretrained=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained)
        self.model.head = nn.Identity()  # remove classification head

        self.decoder = Decoder(in_channs=int(embed_dim * (math.pow(2, len(depths) - 1))),
                               output_channels=output_channels,
                               patch_size = patch_size,
                               )

    def forward(self, x):
        self.stage_outputs = []
        B, C, H, W = x.shape
        x = self.model.patch_embed(x)
        # print(x.shape)
        x = self.model.pos_drop(x)
        # print(x.shape)

        for stage in self.model.layers:
            for blk in stage.blocks:
                x = blk(x)

            # Check if current stage has a downsampling layer before storing and downsampling
            if stage.downsample is not None:
                self.stage_outputs.append(x)  # Store output before downsampling
                x = stage.downsample(x)  # Downsample

        x = self.model.norm(x)
        # print(x.shape)
        # x = self.model.head(x)
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths) - 1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths) - 1)), self.last_stage_dim)
        # print('after reshape:', x.shape)
        # Permute the tensor dimensions to make it compatible with the decoder
        x = x.permute(0, 3, 1, 2)
        # print('after permute:', x.shape)

        x = self.decoder(x, self.stage_outputs)

        return x


class HybridSwinT(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=3, cnn_channels=[16, 32, 64], **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))
        self.cnn_depths = cnn_channels

        '''
        # Add a block of convolutional layers
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
        )
        '''
        # Customizable depths for the CNN module
        layers = []
        in_channels = in_chans
        for channel in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half
            ])
            in_channels = channel
        self.cnn_block = nn.Sequential(*layers)

        # create the Swin Transformer model
        self.model = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=64,  # The input channels for the SwinTransformer is now 64
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.model.head = nn.Identity()  # remove classification head

        self.decoder = Decoder_hbrid(in_channs=int(embed_dim * (math.pow(2, len(depths) - 1))),
                               output_channels=output_channels,
                               patch_size=patch_size,
                               )

    def forward(self, x):
        self.stage_outputs = []
        # Pass the input through the cnn_block first
        # x = self.cnn_block(x)
        for layer in self.cnn_block:
            x = layer(x)
            print(x.shape)
            if isinstance(layer, nn.MaxPool2d):
                self.stage_outputs.append(x)

        B, C, H, W = x.shape

        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for stage in self.model.layers:
            for blk in stage.blocks:
                x = blk(x)
                print(x.shape)
            if stage.downsample is not None:
                self.stage_outputs.append(x)
                x = stage.downsample(x)

        x = self.model.norm(x)
        x = x.reshape(B, self.img_size[0] // int(self.patch_size * math.pow(2, len(self.depths) - 1)),
                      self.img_size[1] // int(self.patch_size * math.pow(2, len(self.depths) - 1)), self.last_stage_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x, self.stage_outputs)

        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class SwinUnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=10, ngf=64, norm_layer=nn.BatchNorm2d,
                 norm_layer_swinT=nn.LayerNorm, use_dropout=False,
                 img_size=1024, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2):
        super(SwinUnetGenerator, self).__init__()

        # UNet branch
        # Downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(num_downs):
            in_channels = input_nc if i == 0 else min(ngf * (2 ** (i - 1)), ngf * 8)
            out_channels = min(ngf * (2 ** i), ngf * 8)
            is_innermost = (i == num_downs - 1)  # Check if it's the innermost layer
            self.down_layers.append(
                self.get_down_layer(in_channels, out_channels, norm_layer, use_dropout, is_innermost))


        # Upsampling layers
        self.up_layers = nn.ModuleList()
        # Define channel configurations for each layer
        channel_configs = [
            (ngf * 8, ngf * 8),  # i == 0 (innermost)
            (ngf * 16, ngf * 8),  # i == 1
            (ngf * 16, ngf * 8),  # i == 2
            (ngf * 16, ngf * 8),  # i == 3
            (ngf * 16, ngf * 8),  # i == 4
            (ngf * 16, ngf * 8),  # i == 5
            (ngf * 16, ngf * 4),  # i == 6
            (ngf * 8, ngf * 2),  # i == 7
            (ngf * 4, ngf),  # i == 8
        ]

        for i in range(num_downs - 1):
            in_channels, out_channels = channel_configs[i]
            self.up_layers.append(self.get_up_layer(in_channels, out_channels, norm_layer, use_dropout))

        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Swin Transformer branch
        self.swinT = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer_swinT
        )
        self.swinT.head = nn.Identity()  # remove classification head

        self.cross_atts = nn.ModuleList([
            Cross_Att(ngf * 8, embed_dim * 2),
            Cross_Att(ngf * 8, embed_dim * 4),
            Cross_Att(ngf * 8, embed_dim * 8),
        ])

        self.cnn_starting_index = math.log2(patch_size)
        self.patch_projector = nn.Conv2d(input_nc, embed_dim, kernel_size=1, stride=1, bias=True)

    def get_down_layer(self, in_channels, out_channels, norm_layer, use_dropout, is_innermost):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        if not is_innermost:  # Only add normalization layer if it's not the innermost layer
            layers.append(norm_layer(out_channels))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def get_up_layer(self, in_channels, out_channels, norm_layer, use_dropout):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            norm_layer(out_channels)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    @staticmethod
    def patch_partition(x, patch_size, overlap_rate):
        stride_h = int(patch_size[0] * (1 - overlap_rate))
        stride_w = int(patch_size[1] * (1 - overlap_rate))
        stride = (stride_h, stride_w)

        B, C, H, W = x.size()

        # Padding might be needed if the defined stride doesn't perfectly divide the height and width
        pad_H = (stride[0] - H % stride[0]) % stride[0]
        pad_W = (stride[1] - W % stride[1]) % stride[1]

        x = F.pad(x, (0, 0, 0, pad_H, 0, pad_W), mode='constant', value=0)

        I_h = torch.arange(0, H + pad_H - patch_size[0] + 1, stride[0]).long()
        I_w = torch.arange(0, W + pad_W - patch_size[1] + 1, stride[1]).long()

        patches = torch.stack([x[:, :, i:i + patch_size[0], j:j + patch_size[1]] for i in I_h for j in I_w])
        patches = patches.permute(1, 0, 2, 3, 4).reshape(B, -1, C * patch_size[0] * patch_size[1])

        return patches

    def forward(self, x):
        x0 = x
        unet_features = []
        swinT_features = []
        skip_connections = []

        # unet processing
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
            if x.shape[2] == 16 or x.shape[2] == 8 or x.shape[2] == 4:
                unet_features.append(x)
                skip_connections[-1] = unet_features[-1]
        skip_connections = skip_connections[:-1] # Remove the last

        # SwinT processing
        B, C, H, W = x0.shape
        print('img shape: ', B, C, H, W)
        x2 = self.swinT.patch_embed(x0)
        x2 = self.swinT.pos_drop(x2)
        for stage in self.swinT.layers:
            for blk in stage.blocks:
                x2 = blk(x2)
            if stage.downsample is not None:
                x2 = stage.downsample(x2)
                h_w_dim = int((x2.shape[1]) ** 0.5)
                print('h_w_dim:', h_w_dim)
                swinT_features.append(x2.view(x2.shape[0], h_w_dim, h_w_dim,
                                              x2.shape[2]).permute(0, 3, 1, 2))

        # Apply Cross Attention at each scale
        for i in range(len(self.cross_atts)):
            #unet_features[i], swinT_features[i] = self.cross_atts[i](unet_features[i], swinT_features[i])
            unet_features[i] = self.cross_atts[i](unet_features[i], swinT_features[i])
        skip_connections[int(self.cnn_starting_index)] = unet_features[0]
        skip_connections[int(self.cnn_starting_index + 1)] = unet_features[1]
        skip_connections[int(self.cnn_starting_index + 2)] = unet_features[2]
        skip_connections = skip_connections[::-1]

        for up_layer, skip in zip(self.up_layers, skip_connections):
            x = up_layer(x)
            x = torch.cat([x, skip], dim=1)

        return self.final_layer(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Cross_Att(nn.Module):
    def __init__(self, dim_cnn, dim_swinT):
        super().__init__()
        self.transformer_unet = Transformer(dim=dim_cnn, depth=1, heads=3, dim_head=32, mlp_dim=128)
        # self.transformer_swinT = Transformer(dim=dim_swinT, depth=1, heads=1, dim_head=64, mlp_dim=256)
        # self.norm_unet = nn.LayerNorm(dim_cnn)
        self.norm_swinT = nn.LayerNorm(dim_swinT)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_unet = nn.Linear(dim_cnn, dim_swinT)
        self.linear_swinT = nn.Linear(dim_swinT, dim_cnn)
        self.gate = nn.Sequential(
            nn.Conv2d(dim_cnn, dim_cnn, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feature, swinT_feature):
        original_cnn_feature = cnn_feature.clone()
        gate_values = self.gate(original_cnn_feature)
        b_u, c_u, h_u, w_u = cnn_feature.shape
        cnn_feature = cnn_feature.reshape(b_u, c_u, -1).permute(0, 2, 1)
        b_s, c_s, h_s, w_s = swinT_feature.shape
        swinT_feature = swinT_feature.reshape(b_s, c_s, -1).permute(0, 2, 1)
        swinT_t = torch.flatten(self.avgpool(self.norm_swinT(swinT_feature).transpose(1,2)), 1)
        swinT_t = self.linear_swinT(swinT_t).unsqueeze(1)
        cnn_feature = self.transformer_unet(torch.cat([swinT_t, cnn_feature],dim=1))[:, 1:, :]
        cnn_feature = cnn_feature.permute(0, 2, 1).reshape(b_u, c_u, h_u, w_u)
        cnn_feature_output = gate_values * cnn_feature + (1 - gate_values) * original_cnn_feature

        return cnn_feature_output


class ResnetGeneratorSwinT(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',
                 norm_layer_swinT=nn.LayerNorm, img_size=1024, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2
                 ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorSwinT, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # Initial layers
        self.initial_layers = self.init_layers(input_nc, ngf, norm_layer, use_bias)

        # Downsampling layers
        self.downsampling_layers = self.downsample_layers(ngf, norm_layer, use_bias)

        # ResNet blocks
        self.resnet_blocks = self.resnet_blocks_layers(ngf, n_blocks, padding_type, norm_layer, use_dropout,
                                                       use_bias)

        # Swin Transformer branch
        self.swinT = timm.models.swin_transformer.SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer_swinT
        )
        self.swinT.head = nn.Identity()  # remove classification head

        self.cross_atts = nn.ModuleList([
            GatedCrossAttention(cnn_channels=128, swinT_channels=192, upsample_factor=int(math.log2(patch_size))),
            GatedCrossAttention(cnn_channels=256, swinT_channels=384, upsample_factor=int(math.log2(patch_size))),
            GatedCrossAttention(cnn_channels=512, swinT_channels=768, upsample_factor=int(math.log2(patch_size)))
        ])

        self.patch_projector = nn.Conv2d(input_nc, embed_dim, kernel_size=1, stride=1, bias=True)

        # Upsampling layers
        self.upsampling_layers = self.upsample_layers(ngf, norm_layer, use_bias)

        # Final layers
        self.final_layers = self.final_layers_func(ngf, output_nc)

    def init_layers(self, input_nc, ngf, norm_layer, use_bias):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        return nn.Sequential(*layers)

    def downsample_layers(self, ngf, norm_layer, use_bias):
        n_downsampling = 3
        layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            block = nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            )
            layers.append(block)
        return layers

    def resnet_blocks_layers(self, ngf, n_blocks, padding_type, norm_layer, use_dropout, use_bias):
        mult = 2 ** 3  # 3 for n_downsampling
        layers = []
        for i in range(n_blocks):
            layers.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias))
        return nn.Sequential(*layers)

    def upsample_layers(self, ngf, norm_layer, use_bias):
        n_downsampling = 3
        layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            block = nn.Sequential(
                nn.ConvTranspose2d(ngf * mult * 2,  # Multiply by 2 to account for concatenation
                                   int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            )
            layers.append(block)
        return layers

    def final_layers_func(self, ngf, output_nc):
        return nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.initial_layers(input)

        # Downsample while storing the intermediate outputs
        downsampled_features = []
        for down_layer in self.downsampling_layers:
            x = down_layer(x)
            downsampled_features.append(x)

        x = self.resnet_blocks(x)

        # SwinT branch
        swinT_features = []
        # SwinT processing
        B, C, H, W = input.shape
        print('input shape: ', B, C, H, W)
        x2 = self.swinT.patch_embed(input)
        x2 = self.swinT.pos_drop(x2)
        for stage in self.swinT.layers:
            for blk in stage.blocks:
                x2 = blk(x2)
            if stage.downsample is not None:
                x2 = stage.downsample(x2)
                h_w_dim = int((x2.shape[1]) ** 0.5)
                swinT_features.append(x2.view(x2.shape[0], h_w_dim, h_w_dim,
                                              x2.shape[2]).permute(0, 3, 1, 2))

        # Apply Cross Attention at each scale
        for i in range(len(self.cross_atts)):
            downsampled_features[i] = self.cross_atts[i](downsampled_features[i], swinT_features[i])

        # Upsample with concatenation
        for up_layer, feature in zip(self.upsampling_layers, reversed(downsampled_features)):
            x = torch.cat([x, feature], dim=1)
            x = up_layer(x)

        x = self.final_layers(x)

        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class GatedCrossAttention(nn.Module):
    def __init__(self, cnn_channels, swinT_channels, num_heads=8, k=0.003, upsample_factor=5):
        super(GatedCrossAttention, self).__init__()

        self.swinT_transform = nn.Conv2d(swinT_channels, cnn_channels, kernel_size=1)

        self.attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)

        self.gate = nn.Sequential(
            nn.Conv2d(cnn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Predefine the upsampling blocks based on the upsample_factor
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(cnn_channels, cnn_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
            ) for _ in range(upsample_factor)
        ])
        self.k_raw = nn.Parameter(torch.tensor([math.log(0.01 / (1 - 0.01))]))

    def forward(self, downsampling_features, swinT_features):
        # Transform channel dimensions to common_channels
        swinT_features = self.swinT_transform(swinT_features)

        for upsample_block in self.upsample_blocks:
            swinT_features = upsample_block(swinT_features)

        # Calculate gate values
        gate_values = self.gate(downsampling_features)

        B, C, H, W = gate_values.size()
        k = torch.sigmoid(self.k_raw)
        currentk = int(H * W * k)
        print(currentk)

        # Flatten and permute for attention module
        down_features_flat = downsampling_features.flatten(2).permute(2, 0, 1)
        swinT_features_flat = swinT_features.flatten(2).permute(2, 0, 1)

        # Select top-k activations to apply attention
        _, top_indices = torch.topk(gate_values.view(gate_values.size(0), -1), k=currentk, dim=1)

        down_features_subset = torch.index_select(down_features_flat, 0, top_indices.view(-1))
        swinT_features_subset = torch.index_select(swinT_features_flat, 0, top_indices.view(-1))

        # Apply attention only on the subset
        attended_features_subset, _ = self.attention(down_features_subset, swinT_features_subset, swinT_features_subset)

        # Scatter back the attended values to original size tensor
        attended_features = down_features_flat.clone()
        attended_features.index_copy_(0, top_indices.view(-1), attended_features_subset)

        # Reshape to [B, C, H, W]
        attended_features = attended_features.permute(1, 2, 0).view_as(downsampling_features)

        # Combine the attended features and original features using the gate
        out = gate_values * attended_features + (1 - gate_values) * downsampling_features

        return out