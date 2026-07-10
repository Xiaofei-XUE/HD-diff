"""HD-Diff model definition for the public source release.

This file exposes the model composition used by HD-Diff while intentionally
omitting dataset loading, training loops, experiment scripts, checkpoints, and
project-specific runtime paths.
"""

import torch
import torch.nn as nn

from guided_diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from unet.basic_unet_Segmentation import BasicUNetDe
from unet.Encoder_S_DFM_full_BEC_CEC_layerwise import (
    SDFMFullLayerwiseBoundaryCoreEnhanceEncoder,
)


NUMBER_MODALITY = 4
NUMBER_TARGETS = 3
FEATURES = [64, 64, 128, 256, 512, 64]
ROI_SIZE = [96, 96, 96]


class DiffUNet(nn.Module):
    def __init__(
        self,
        boundary_scales=(0, 1),
        deep_et_scales=(3, 4),
        et_edge_boost=1.0,
        et_core_boost=1.0,
        rsg_guidance_start_ratio=0.0,
        rsg_guidance_update_every=2,
        rsg_guidance_momentum=0.8,
        infer_ddim_steps=10,
    ):
        super().__init__()
        self.rsg_guidance_start_ratio = rsg_guidance_start_ratio
        self.rsg_guidance_update_every = rsg_guidance_update_every
        self.rsg_guidance_momentum = rsg_guidance_momentum

        self.embed_model = SDFMFullLayerwiseBoundaryCoreEnhanceEncoder(
            spatial_dims=3,
            in_channels_per_modality=1,
            num_modalities=NUMBER_MODALITY,
            num_targets=NUMBER_TARGETS,
            features=FEATURES,
            boundary_scales=boundary_scales,
            deep_et_scales=deep_et_scales,
            et_edge_boost=et_edge_boost,
            et_core_boost=et_core_boost,
        )
        self.model = BasicUNetDe(
            3,
            NUMBER_MODALITY + NUMBER_TARGETS,
            NUMBER_TARGETS,
            FEATURES,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [infer_ddim_steps]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def recursive_self_guided_ddim_sample(self, image, init_label=None):
        device = image.device
        shape = (image.shape[0], NUMBER_TARGETS, *ROI_SIZE)
        img = torch.randn(*shape, device=device)
        embeddings = self.embed_model(image, label=init_label)
        guided_mask = init_label
        pred_xstart = None
        num_steps = self.sample_diffusion.num_timesteps
        guidance_start = int(num_steps * self.rsg_guidance_start_ratio)

        for step_idx, i in enumerate(list(range(num_steps))[::-1]):
            t = torch.tensor([i] * shape[0], device=device)
            out = self.sample_diffusion.ddim_sample(
                self.model,
                img,
                t,
                model_kwargs={"image": image, "embeddings": embeddings},
            )
            pred_xstart = out["pred_xstart"]
            if step_idx >= guidance_start and step_idx % self.rsg_guidance_update_every == 0:
                soft_mask = torch.sigmoid(pred_xstart).detach()
                guided_mask = (
                    soft_mask if guided_mask is None
                    else self.rsg_guidance_momentum * guided_mask
                    + (1.0 - self.rsg_guidance_momentum) * soft_mask
                )
                embeddings = self.embed_model(image, label=guided_mask.clamp(0.0, 1.0))
            img = out["sample"]

        return pred_xstart if pred_xstart is not None else img

    def forward(self, image=None, x=None, pred_type=None, step=None, label=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, _ = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        if pred_type == "denoise":
            embeddings = self.embed_model(image, label=label)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        if pred_type == "ddim_sample":
            return self.recursive_self_guided_ddim_sample(image)

        raise ValueError(f"Unsupported pred_type: {pred_type}")
