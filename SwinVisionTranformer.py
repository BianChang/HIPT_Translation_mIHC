import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from timm.models.swin_transformer import SwinTransformerBlock
import math
from torch.utils.checkpoint import checkpoint
from timm.models.swin_transformer import PatchMerging


class Decoder(nn.Module):
    def __init__(self, in_channs, output_channels, img_size, patch_size):
        super().__init__()

        self.upsample1 = nn.ConvTranspose2d(in_channs, in_channs // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channs, in_channs // 2, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channs // 2, in_channs // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channs // 2, in_channs// 4, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channs // 4, in_channs // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channs // 4, in_channs // 8, kernel_size=3, padding=1)

        self.upsample4 = nn.ConvTranspose2d(in_channs // 8, in_channs // 8, kernel_size=6, stride=4, padding=1)
        self.upsample5 = nn.ConvTranspose2d(in_channs // 8, output_channels, kernel_size=6, stride=4, padding=1)


    def forward(self, x, stage_outputs):
        x = self.upsample1(x)
        x = torch.cat((x, stage_outputs[-1]), dim=1)
        x = self.conv1(x)

        x = self.upsample2(x)
        x = torch.cat((x, stage_outputs[-2]), dim=1)
        x = self.conv2(x)

        x = self.upsample3(x)
        x = torch.cat((x, stage_outputs[-3]), dim=1)
        x = self.conv3(x)

        x = self.upsample4(x)
        x = self.upsample5(x)

        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=[1024, 1024], patch_size=16, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=3, window_size=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=4, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.num_patches_sqrt = int(
            math.sqrt(self.num_patches))  # compute the square root of num_patches and cast to int

        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, embed_dim))
        nn.init.kaiming_uniform_(self.pos_embed, a=math.sqrt(5))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage_outputs = []

        self.blocks_and_merging = nn.ModuleList([])

        for i in range(len(depths)):
            stage_blocks = [
                SwinTransformerBlock(
                    dim=embed_dim, input_resolution=(self.num_patches_sqrt, self.num_patches_sqrt),
                    num_heads=num_heads, window_size=window_size, shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate, norm_layer=norm_layer)
                for _ in range(depths[i])
            ]

            self.blocks_and_merging.append(nn.Sequential(*stage_blocks))

            if i < len(depths) - 1:  # Don't add patch merging after the last stage
                patch_merging = PatchMerging(input_resolution=(self.num_patches_sqrt, self.num_patches_sqrt),
                                             dim=embed_dim, norm_layer=norm_layer)
                self.blocks_and_merging.append(patch_merging)
        '''
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                SwinTransformerBlock(
                    dim=embed_dim, input_resolution=(self.num_patches_sqrt, self.num_patches_sqrt), num_heads=num_heads, window_size=window_size, shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate, norm_layer=norm_layer)
                for _ in range(depth)])
            for depth in depths])
        '''

        self.decoder = Decoder(in_channs=embed_dim // (math.pow(2, len(depths))-1), output_channels=output_channels,
                               img_size=img_size, patch_size=patch_size)

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
        # Apply patch embedding to convert the input image into a sequence of flattened patches
        print("input shape:", x.shape)
        x = self.patch_embed(x)

        # Extract the batch size (B), channels (C), height (H), and width (W) from the input tensor
        B, C, H, W = x.shape
        # Calculate the number of patches by dividing the height and width by the patch size
        # num_patches_2 = (H // self.patch_size) * (W // self.patch_size)

        # Reshape the input tensor to create patches
        x = x.reshape(B, C, self.patch_size, H // self.patch_size, self.patch_size, W // self.patch_size)
        # Permute and reshape the tensor to obtain a sequence of flattened patches
        # x = x.permute(0, 3, 5, 1, 2, 4).reshape(B, num_patches, C * self.patch_size ** 2)
        x = x.permute(0, 3, 5, 1, 2, 4).reshape(B, self.num_patches, C)


        # Add positional encoding to the patch embeddings
        x = x + self.pos_embed
        print('x+x_embed: ', x.shape)
        # Apply dropout to the patch embeddings (prevent overfitting)
        x = self.pos_drop(x)

        # Process the patch embeddings through the Swin Transformer blocks
        for i, layer in enumerate(self.blocks_and_merging):
            x = layer(x)
            if i + 1 < len(self.blocks_and_merging) and isinstance(self.blocks_and_merging[i + 1], PatchMerging):
                # Save the output of each stage before the patch merging in self.stage_outputs
                self.stage_outputs.append(x)

        '''
        for stage in self.blocks:
            x = stage(x)
            print(x.shape)
        '''

        # Reshape the output tensor to obtain the image features in the original spatial dimensions
        x = x.reshape(B, self.img_size[0] // (self.patch_size * math.pow(2, len(self.depths)-1)) ,
                      self.img_size[1] // (self.patch_size * math.pow(2, len(self.depths)-1)), self.embed_dim)
        # Permute the tensor dimensions to make it compatible with the decoder
        x = x.permute(0, 3, 1, 2)

        # Add skip connections in the decoder
        # Add skip connections in the decoder
        x = self.decoder(x, self.stage_outputs)
        # Reshape the output tensor to obtain the output image with the correct dimensions
        x = x.view(B, -1, self.img_size[0], self.img_size[1])


        return x
