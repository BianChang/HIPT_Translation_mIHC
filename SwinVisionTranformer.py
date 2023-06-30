import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from timm.models.swin_transformer import SwinTransformerBlock
import math
from torch.utils.checkpoint import checkpoint
from timm.models.swin_transformer import PatchMerging, PatchEmbed
import timm
import functools
from torch.nn import init


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

        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        upsampling_factor = int(math.log(patch_size, 2))+2
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
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=3, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size
        self.last_stage_dim = embed_dim * (2 ** (len(depths) - 1))

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
        x = self.cnn_block(x)
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

