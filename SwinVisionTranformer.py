import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from timm.models.swin_transformer import SwinTransformerBlock
import math

class SwinTransformer(nn.Module):
    def __init__(self, img_size=[4096], patch_size=16, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=3, window_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, output_channels=4, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (img_size[0] // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, embed_dim))
        nn.init.kaiming_uniform_(self.pos_embed, a=math.sqrt(5))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                SwinTransformerBlock(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity(),
                    norm_layer=norm_layer)
                for _ in range(depth)])
            for depth in depths])

        '''
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_patches, in_chans * img_size[0] * img_size[0]),
            nn.BatchNorm1d(in_chans * img_size[0] * img_size[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (in_chans, img_size[0], img_size[0])),
            nn.Sequential(
                nn.ConvTranspose2d(in_chans, output_channels, kernel_size=patch_size, stride=patch_size, padding=0),
                nn.Upsample(scale_factor=img_size[0] // (patch_size * patch_size), mode='bilinear', align_corners=True)
            ),
        )
        '''

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chans, output_channels, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.Upsample(scale_factor=img_size[0] // (patch_size * patch_size), mode='bilinear', align_corners=True),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # Apply patch embedding to convert the input image into a sequence of flattened patches
        x = self.patch_embed(x)

        # Extract the batch size (B), channels (C), height (H), and width (W) from the input tensor
        B, C, H, W = x.shape
        # Calculate the number of patches by dividing the height and width by the patch size
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Reshape the input tensor to create patches
        x = x.reshape(B, C, self.patch_size, H // self.patch_size, self.patch_size, W // self.patch_size)
        # Permute and reshape the tensor to obtain a sequence of flattened patches
        x = x.permute(0, 3, 5, 1, 2, 4).reshape(B, num_patches, C * self.patch_size ** 2)

        # Add positional encoding to the patch embeddings
        x = x + self.pos_embed
        # Apply dropout to the patch embeddings (prevent overfitting)
        x = self.pos_drop(x)

        # Process the patch embeddings through the Swin Transformer blocks
        for stage in self.blocks:
            x = stage(x)

        # Reshape the output tensor to obtain the image features in the original spatial dimensions
        x = x.reshape(B, H // self.patch_size, W // self.patch_size, self.embed_dim)
        # Permute the tensor dimensions to make it compatible with the decoder
        x = x.permute(0, 3, 1, 2)

        # Add skip connections in the decoder
        x_skip = x
        x = self.decoder(x)
        x = x + x_skip
        # Reshape the output tensor to obtain the output image with the correct dimensions
        x = x.view(B, -1, H, W)


        return x
