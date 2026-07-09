# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet", "BasicUNetEncoder", "MultiModalUNetEncoder"]


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class BasicUNet(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        embeddings = []

        x0 = self.conv_0(x)
        embeddings.append(x0)

        x1 = self.down_1(x0)
        embeddings.append(x1)

        x2 = self.down_2(x1)
        embeddings.append(x2)

        x3 = self.down_3(x2)
        embeddings.append(x3)

        x4 = self.down_4(x3)
        embeddings.append(x4)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return logits, embeddings


BasicUnet = Basicunet = basicunet = BasicUNet


class BasicUNetEncoder(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return [x0, x1, x2, x3, x4]


class SEBlock3D(nn.Module):
    """
    3D SE模块（替代DCFormer）：用于模态内特征校准 + 跨模态冗余过滤
    核心：通道注意力机制，学习每个通道的重要性权重，抑制冗余特征
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        # 全局平均池化：将3D特征 (B,C,H,W,D) → (B,C,1,1,1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 通道注意力权重学习
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # 全局池化
        y = self.avg_pool(x).view(b, c)
        # 学习注意力权重
        y = self.fc(y).view(b, c, 1, 1, 1)
        # 特征校准：原特征 × 注意力权重
        return x * y.expand_as(x)


class DualStreamAdapter(nn.Module):
    def __init__(self, channel: int, expand_ratio: float = 1.25, reduction: int = 4):
        super().__init__()
        expand_channel = max(8, int(channel * expand_ratio))
        reduce_channel = max(8, channel // reduction)
        self.branch_union = nn.Sequential(
            Convolution(
                3,
                channel,
                expand_channel,
                act=("leakyrelu", {"negative_slope": 0.1}),
                norm=("instance", {"affine": True}),
                bias=False,
                kernel_size=1,
            ),
            Convolution(
                3,
                expand_channel,
                channel,
                act=None,
                norm=("instance", {"affine": True}),
                bias=False,
                kernel_size=1,
            ),
        )
        self.branch_inter = nn.Sequential(
            Convolution(
                3,
                channel,
                reduce_channel,
                act=("leakyrelu", {"negative_slope": 0.1}),
                norm=("instance", {"affine": True}),
                bias=False,
                kernel_size=1,
            ),
            Convolution(
                3,
                reduce_channel,
                channel,
                act=None,
                norm=("instance", {"affine": True}),
                bias=False,
                kernel_size=1,
            ),
        )

    def forward(self, union_feat: torch.Tensor, inter_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        refined_union = union_feat + self.branch_inter(inter_feat)
        refined_inter = inter_feat + self.branch_union(union_feat)
        return refined_union, refined_inter


class DepthwiseScaleConv3D(nn.Sequential):
    def __init__(self, channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )


class PaperMultiScaleFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(8, out_channels // reduction)
        self.pool_scales = (1, 2, 3)
        self.reduce = Convolution(
            3,
            in_channels,
            hidden_channels,
            act=("leakyrelu", {"negative_slope": 0.1}),
            norm=("instance", {"affine": True}),
            bias=False,
            kernel_size=1,
        )
        self.branch_3 = DepthwiseScaleConv3D(hidden_channels, kernel_size=3)
        self.branch_5 = DepthwiseScaleConv3D(hidden_channels, kernel_size=5)
        self.branch_7 = DepthwiseScaleConv3D(hidden_channels, kernel_size=7)
        self.fuse = Convolution(
            3,
            hidden_channels * 3,
            out_channels,
            act=("leakyrelu", {"negative_slope": 0.1}),
            norm=("instance", {"affine": True}),
            bias=False,
            kernel_size=1,
        )
        self.pool_projects = nn.ModuleList([
            Conv["conv", 3](in_channels, out_channels, kernel_size=1, bias=False)
            for _ in self.pool_scales
        ])

    def forward(self, aug_feat: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(aug_feat)
        local_feat = torch.cat(
            [
                self.branch_3(reduced),
                self.branch_5(reduced),
                self.branch_7(reduced),
            ],
            dim=1,
        )
        local_feat = self.fuse(local_feat)

        pooled_feats = []
        for scale, project in zip(self.pool_scales, self.pool_projects):
            pooled = F.adaptive_avg_pool3d(aug_feat, output_size=scale)
            pooled = project(pooled)
            pooled = F.interpolate(pooled, size=local_feat.shape[2:], mode="trilinear", align_corners=False)
            pooled_feats.append(pooled)

        return local_feat + torch.stack(pooled_feats, dim=0).sum(dim=0)


class PaperDFM(nn.Module):
    def __init__(self, channel: int, out_channels: int, num_modalities: int = 4):
        super().__init__()
        self.num_modalities = num_modalities
        self.adapter = DualStreamAdapter(channel)
        aug_channels = (num_modalities + 2) * channel
        self.multi_scale_fusion = PaperMultiScaleFusion(aug_channels, out_channels)

    def forward(self, modal_features: List[torch.Tensor]) -> torch.Tensor:
        stacked_features = torch.stack(modal_features, dim=0)
        union_feat = torch.max(stacked_features, dim=0)[0]
        inter_feat = torch.min(stacked_features, dim=0)[0]
        refined_union, refined_inter = self.adapter(union_feat, inter_feat)
        aug_feat = torch.cat([refined_union, refined_inter] + modal_features, dim=1)
        return self.multi_scale_fusion(aug_feat)


class MultiModalUNetEncoder(nn.Module):
    """
    集成SE模块的多模态UNet编码器：
    1. 4个模态独立通过BasicUNetEncoder提取特征
    2. 每个模态的每个尺度特征通过SE模块做特征校准（过滤冗余）
    3. 跨模态尺度融合（concat/add/mean）
    输入：4通道张量 (B,4,H,W,D) → BraTS4模态
    输出：融合后的多尺度特征 [x0, x1, x2, x3, x4]
    """
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels_per_modality: int = 1,
        num_modalities: int = 4,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        fusion_mode: str = "concat",
        se_reduction: int = 16,  # SE模块的通道缩减率
        dfm_out_channels: Sequence[int] = (64, 64, 128, 256, 512),
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        self.num_modalities = num_modalities
        self.fusion_mode = fusion_mode
        self.features = features
        self.dfm_out_channels = ensure_tuple_rep(dfm_out_channels, 5)

        # 1. 每个模态独立的编码器
        self.encoders = nn.ModuleList([
            BasicUNetEncoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels_per_modality,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout
            ) for _ in range(num_modalities)
        ])

        # 2. 为每个模态的每个尺度特征配置SE模块（特征校准）
        self.se_blocks = nn.ModuleDict()
        for mod_idx in range(num_modalities):
            for scale_idx in range(5):  # 5个尺度：x0~x4
                ch = features[scale_idx]  # 当前尺度的通道数
                self.se_blocks[f"se_mod{mod_idx}_scale{scale_idx}"] = SEBlock3D(ch, se_reduction)

        self.dfm_modules = nn.ModuleList([
            PaperDFM(features[scale_idx], self.dfm_out_channels[scale_idx], num_modalities=num_modalities)
            for scale_idx in range(5)
        ])

    def forward(self, x: torch.Tensor):
        # 步骤1：拆分4通道输入为4个单模态张量 (B,1,H,W,D)
        modality_tensors = [x[:, i:i+1, ...] for i in range(self.num_modalities)]

        # 步骤2：每个模态独立编码 + SE模块特征校准
        modal_features = []
        for mod_idx, mod_tensor in enumerate(modality_tensors):
            # 模态独立编码
            mod_feat = self.encoders[mod_idx](mod_tensor)
            # SE模块校准每个尺度的特征
            calibrated_feat = []
            for scale_idx, feat in enumerate(mod_feat):
                se_block = self.se_blocks[f"se_mod{mod_idx}_scale{scale_idx}"]
                calibrated_feat.append(se_block(feat))
            modal_features.append(calibrated_feat)

        # 步骤3：论文原版 DFM 融合
        fused_features = []
        for scale_idx in range(5):
            scale_feats = [modal_feat[scale_idx] for modal_feat in modal_features]
            fused_features.append(self.dfm_modules[scale_idx](scale_feats))

        return fused_features
