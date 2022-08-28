import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from nnunet.drop import DropPath, trunc_normal_
import math


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        x_ = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + x_
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.mlp(self.norm2(x))
        )
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=768,
        transpose=False,
    ):
        super().__init__()
        patch_size = (
            patch_size
            if isinstance(patch_size, (tuple, list))
            else tuple(3 * [img_size])
        )
        conv = nn.ConvTranspose3d if transpose else nn.Conv3d
        pad = [p // 2 for p in patch_size]
        pad = [p if p >= 0 else 0 for p in pad]
        pad = tuple(pad)
        kwargs = {"output_padding": 1} if transpose else {}
        self.proj = conv(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=pad,
            **kwargs
        )
        self.norm = nn.BatchNorm3d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = self.norm(x)
        return x, H, W, D


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MDBlock(nn.Module):
    def __init__(
        self,
        img_size,
        in_chans,
        embed_chans,
        patch_size=3,
        stride=2,
        num_units=1,
        dropout=0.0,
        mlp_ratio=4,
        path_dropout=0.0,
        norm_layer=nn.LayerNorm,
        transpose=False,
    ):
        super(MDBlock, self).__init__()
        if type(path_dropout) != list:
            path_dropout = num_units * [path_dropout]  # stochastic depth decay rule
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_chans,
            transpose=transpose,
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=embed_chans,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    drop_path=path_dropout[i],
                )
                for i in range(num_units)
            ]
        )
        self.norm = norm_layer(embed_chans)

    def forward(self, x, x_skip=None):
        B = x.shape[0]
        x, H, W, D = self.patch_embed(x)
        for blk in self.layers:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return x
