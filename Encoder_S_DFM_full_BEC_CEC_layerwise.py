"""
改进版 BEC/CEC 编码器
=====================

相比原版，本文件做了三项针对 ET 的改进：

  ① boundary_scales 默认从 (0,1,2) 改为 (0,1)
     scale 2（24³）改走 CEC，ET 在 24³ 仍有 50-200 体素，CEC 能学到有效信号。

  ② CEC 里 label 下采样从 nearest-interpolate 改为逐级 max_pool3d
     原来 96³ → 6³ 用 nearest 插值，ET 体素极易被跳过（概率约 99.97% 丢失）。
     改用 max_pool 链：任意一个 ET 体素都能传播到输出格，信号不再消失。

  ③ 新增 ETGlobalContextInjector，在深层 scale 3/4 做通道级 ET 全局注入
     scale 3（12³）和 scale 4（6³）里 ET 体素数 < 1，无法靠空间 mask 工作。
     改为从 label 提取 [ET 是否存在, ET 体积比] 两个全局统计，
     经 MLP 生成通道注意力权重，注入到深层特征，无需空间对齐。
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep

from unet.Encoder_S_DFM_paper_fullDFM import MultiModalUNetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 原版模块（保持不变）
# ─────────────────────────────────────────────────────────────────────────────

class LabelBoundaryMask3D(nn.Module):
    """Build scale-aligned 3D boundary masks from GT labels or predicted masks."""

    def forward(self, label: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        label_i = F.interpolate(label.float(), size=target_size, mode="nearest")
        edge_masks = []
        for c in range(label_i.shape[1]):
            single = label_i[:, c:c + 1]
            dx = F.pad(torch.abs(single[:, :, 1:] - single[:, :, :-1]),   (0, 0, 0, 0, 1, 0))
            dy = F.pad(torch.abs(single[:, :, :, 1:] - single[:, :, :, :-1]),  (0, 0, 1, 0, 0, 0))
            dz = F.pad(torch.abs(single[:, :, :, :, 1:] - single[:, :, :, :, :-1]), (1, 0, 0, 0, 0, 0))
            edge_masks.append(torch.clamp(dx + dy + dz, 0.0, 1.0))
        return torch.cat(edge_masks, dim=1)


class BoundaryAttention3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        return self.attn(torch.cat([feat, edge_mask], dim=1))


class CoreAttention3D(nn.Module):
    def __init__(self, in_channels: int, num_targets: int = 3, hidden_channels: int = 64, dropout: float = 0.0):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels * 2 + num_targets, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, core_mask: torch.Tensor) -> torch.Tensor:
        global_feat = self.global_pool(feat).expand_as(feat)
        return self.attn(torch.cat([feat, global_feat, core_mask], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# ③ 新增：深层 ET 全局上下文注入器
# ─────────────────────────────────────────────────────────────────────────────

class ETGlobalContextInjector(nn.Module):
    """
    专为 scale 3/4（6³/12³）设计的 ET 全局信号注入器。

    ET 在深层特征图（6³）里平均不足 1 个体素，空间 mask 完全失效。
    本模块改为从原始 label 提取两个全局统计量：
      - et_exists：该 batch 内 ET 是否存在（0/1）
      - et_ratio ：ET 体素占 patch 体积的比例（连续值）
    经 MLP 映射为通道注意力权重，对深层特征做通道级调制。
    无需空间对齐，完全绕开分辨率消失问题。

    label=None 时直接跳过（推理无条件阶段不注入）。
    """

    def __init__(self, feat_channels: int, et_channel: int = 2):
        super().__init__()
        self.et_channel = et_channel
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, feat_channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, label: Optional[torch.Tensor]) -> torch.Tensor:
        if label is None:
            return feat
        et    = label[:, self.et_channel:self.et_channel + 1].float()
        total = float(et.shape[2] * et.shape[3] * et.shape[4])
        et_sum    = et.sum(dim=(1, 2, 3, 4))
        et_exists = (et_sum > 0).float()                  # (B,)
        et_ratio  = et_sum / (total + 1e-6)               # (B,)
        ctx    = torch.stack([et_exists, et_ratio], dim=1)          # (B, 2)
        weight = self.mlp(ctx).view(-1, feat.shape[1], 1, 1, 1)     # (B, C, 1, 1, 1)
        return feat * weight


# ─────────────────────────────────────────────────────────────────────────────
# 主编码器
# ─────────────────────────────────────────────────────────────────────────────

class SDFMFullLayerwiseBoundaryCoreEnhanceEncoder(nn.Module):
    """
    Full DFM + layer-wise BEC/CEC encoder（改进版）。

    尺度分工：
      scale 0 (96³)  → BEC：高分辨率边界增强
      scale 1 (48³)  → BEC：边界细节保留
      scale 2 (24³)  → CEC：ET 在此仍有 50-200 体素，空间 mask 有效
      scale 3 (12³)  → CEC + ET 全局注入：空间 mask 弱，全局补充
      scale 4  (6³)  → CEC + ET 全局注入：空间 mask 基本失效，全局主导
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels_per_modality: int = 1,
        num_modalities: int = 4,
        num_targets: int = 3,
        features: Sequence[int] = (64, 64, 128, 256, 512, 64),
        dfm_out_channels: Sequence[int] = (64, 64, 128, 256, 512),
        dropout: float = 0.0,
        boundary_scales: Sequence[int] = (0, 1),      # ① 改：原来 (0,1,2)
        deep_et_scales: Sequence[int]  = (3, 4),      # ③ 新增：深层全局注入
        et_edge_boost: float = 1.0,
        et_core_boost: float = 1.0,
    ):
        super().__init__()
        self.num_targets      = num_targets
        self.tc_channel       = 0
        self.et_channel       = min(2, num_targets - 1)
        self.et_edge_boost    = et_edge_boost
        self.et_core_boost    = et_core_boost
        self.dfm_out_channels = ensure_tuple_rep(dfm_out_channels, 5)
        self.boundary_scales  = set(boundary_scales)
        self.deep_et_scales   = set(deep_et_scales)

        self.dfm_encoder = MultiModalUNetEncoder(
            spatial_dims=spatial_dims,
            in_channels_per_modality=in_channels_per_modality,
            num_modalities=num_modalities,
            features=features,
            dfm_out_channels=self.dfm_out_channels,
        )
        self.boundary_mask = LabelBoundaryMask3D()

        self.boundary_attns = nn.ModuleList([
            BoundaryAttention3D(ch + num_targets, ch, dropout=dropout)
            for ch in self.dfm_out_channels
        ])
        self.core_attns = nn.ModuleList([
            CoreAttention3D(ch, num_targets=num_targets, dropout=dropout)
            for ch in self.dfm_out_channels
        ])
        self.boundary_residual_gates = nn.Parameter(torch.zeros(len(self.dfm_out_channels)))
        self.core_residual_gates     = nn.Parameter(torch.zeros(len(self.dfm_out_channels)))

        # ③ 深层 ET 全局注入器（scale 3, 4）
        self.et_injectors = nn.ModuleDict({
            str(s): ETGlobalContextInjector(self.dfm_out_channels[s], et_channel=self.et_channel)
            for s in self.deep_et_scales
        })

    # ── 伪标签（无 label 时使用）────────────────────────────────────────────
    def _pseudo_label_from_feature(self, feat: torch.Tensor) -> torch.Tensor:
        score = torch.mean(feat, dim=1, keepdim=True)
        score = (score > score.mean(dim=(2, 3, 4), keepdim=True)).float()
        return score.repeat(1, self.num_targets, 1, 1, 1)

    def _pseudo_core_from_feature(self, feat: torch.Tensor) -> torch.Tensor:
        score = torch.mean(feat, dim=1, keepdim=True)
        threshold = score.mean(dim=(2, 3, 4), keepdim=True) + score.std(dim=(2, 3, 4), keepdim=True)
        core = (score >= threshold).float()
        return core.repeat(1, self.num_targets, 1, 1, 1)

    # ── ① 边缘权重 ──────────────────────────────────────────────────────────
    def _edge_local_weight(self, edge_mask: torch.Tensor) -> torch.Tensor:
        any_edge = edge_mask.max(dim=1, keepdim=True)[0]
        any_band = F.max_pool3d(any_edge, kernel_size=3, stride=1, padding=1)
        et_edge  = edge_mask[:, self.et_channel:self.et_channel + 1]
        et_band  = F.max_pool3d(et_edge, kernel_size=5, stride=1, padding=2)
        return torch.clamp(any_band + self.et_edge_boost * et_band, 0.0, 2.0)

    def _core_local_weight(self, core_mask: torch.Tensor) -> torch.Tensor:
        tc_mask = core_mask[:, self.tc_channel:self.tc_channel + 1]
        et_mask = core_mask[:, self.et_channel:self.et_channel + 1]
        tc_band = F.max_pool3d(tc_mask, kernel_size=5, stride=1, padding=2)
        et_band = F.max_pool3d(et_mask, kernel_size=5, stride=1, padding=2)
        return torch.clamp(tc_band + self.et_core_boost * et_band, 0.0, 2.0)

    # ── ② 改进：label 下采样用 max_pool 链，保留 ET 信号 ─────────────────────
    def _max_downsample_to(self, label: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        """
        逐级 max_pool3d 下采样 label 到目标尺寸。

        原来用 F.interpolate(..., mode='nearest')：
          96³ → 6³ 时每个输出格只取 1 个输入体素，ET 以 (6/96)³ ≈ 0.02% 的概率被保留。

        现在用 max_pool 链：
          每一步 /2，ET 所在的 2³ 邻域取最大值，任意 ET 体素都会传播到输出格。
          ET 从"几乎消失"变为"只要存在就能被感知"。
        """
        x = label.float()
        th, tw, td = target_size
        # 逐步 /2 直到再减一次就会过小
        while x.shape[2] > th * 2 or x.shape[3] > tw * 2 or x.shape[4] > td * 2:
            x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
        # 若还有余量差，用 nearest 微调
        if tuple(x.shape[2:]) != (th, tw, td):
            x = F.interpolate(x, size=(th, tw, td), mode="nearest")
        return x

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, image: torch.Tensor, label: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        fused_features   = self.dfm_encoder(image)
        enhanced_features = []

        for scale_idx, feat in enumerate(fused_features):

            if scale_idx in self.boundary_scales:
                # ── BEC：scale 0 (96³), scale 1 (48³) ──────────────────────
                if label is None:
                    scale_label = self._pseudo_label_from_feature(feat)
                    edge_mask   = self.boundary_mask(scale_label, feat.shape[2:])
                else:
                    edge_mask = self.boundary_mask(label, feat.shape[2:])

                boundary_attn = self.boundary_attns[scale_idx](feat, edge_mask)
                edge_weight   = self._edge_local_weight(edge_mask)
                gate          = torch.tanh(self.boundary_residual_gates[scale_idx])
                enhanced_feat = feat + gate * edge_weight * feat * boundary_attn

            else:
                # ── CEC：scale 2 (24³), 3 (12³), 4 (6³) ────────────────────
                if label is None:
                    core_mask = self._pseudo_core_from_feature(feat)
                else:
                    # ② max_pool 链下采样，保留 ET 信号
                    core_mask = self._max_downsample_to(label, feat.shape[2:])

                core_attn     = self.core_attns[scale_idx](feat, core_mask)
                core_weight   = self._core_local_weight(core_mask)
                gate          = torch.tanh(self.core_residual_gates[scale_idx])
                enhanced_feat = feat + gate * core_weight * feat * core_attn

                # ③ 深层全局 ET 注入（scale 3, 4）
                if scale_idx in self.deep_et_scales:
                    enhanced_feat = self.et_injectors[str(scale_idx)](enhanced_feat, label)

            enhanced_features.append(enhanced_feat)

        return enhanced_features
