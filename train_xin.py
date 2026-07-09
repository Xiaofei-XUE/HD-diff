"""
train_xin.py
============

Single-stage training script based on train_v5.py.
Changes:
  1. Load dataset/dataset.py.
  2. Train only fold=0.
  3. Remove stage-2 training.
  4. Use loss = Dice + BCE + MSE with equal weights.
  5. Use fixed condition probabilities: GT=0.2, noisy=0.2, self-guided=0.4, none=0.2.
  6. Use single-pass DDIM inference.
  7. Use sliding-window inference with overlap=0.5.
  8. Remove post-training two-pass checkpoint re-evaluation.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dataset.dataset import get_loader_brats

from guided_diffusion.gaussian_diffusion import (
    LossType, ModelMeanType, ModelVarType, get_named_beta_schedule,
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism

from unet.basic_unet_Segmentation import BasicUNetDe
from unet.Encoder_S_DFM_full_BEC_CEC_layerwise import (
    SDFMFullLayerwiseBoundaryCoreEnhanceEncoder,
)


DEFAULT_DATA_DIR = "./data/MICCAI_BraTS2020_TrainingData/"
DEFAULT_LOGDIR = "./logs_xin/"

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


class XinTrainer(Trainer):
    def __init__(
        self,
        env_type,
        max_epochs,
        batch_size,
        device="cpu",
        val_every=20,
        num_gpus=1,
        logdir="./logs/",
        master_ip="localhost",
        master_port=17840,
        training_script="train_xin.py",
        lr=2e-5,
        warmup_epochs=5,
        infer_ddim_steps=10,
        num_workers=8,
    ):
        super().__init__(
            env_type,
            max_epochs,
            batch_size,
            device,
            val_every,
            num_gpus,
            logdir,
            master_ip,
            master_port,
            training_script,
        )
        self.window_infer = SlidingWindowInferer(
            roi_size=ROI_SIZE,
            sw_batch_size=4,
            overlap=0.5,
        )
        self.model = DiffUNet(infer_ddim_steps=infer_ddim_steps)
        self.model_save_path = os.path.join(logdir, "model")
        self.best_mean_dice = 0.0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=8e-4
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.mse = nn.MSELoss()
        self.train_num_workers = num_workers
        self.scaler = GradScaler()
        self.auto_optim = False

        self.gt_prob = 0.20
        self.noisy_prob = 0.20
        self.self_prob = 0.40
        self.none_prob = 0.20

    def _make_noisy_condition(self, label):
        noisy = label.clone()
        keep = (torch.rand_like(noisy) > 0.10).float()
        noisy = noisy * keep
        if torch.rand(1, device=noisy.device).item() < 0.5:
            noisy = F.max_pool3d(noisy, kernel_size=3, stride=1, padding=1)
        if torch.rand(1, device=noisy.device).item() < 0.5:
            noisy = F.avg_pool3d(noisy, kernel_size=3, stride=1, padding=1)
        noisy = noisy + 0.05 * torch.randn_like(noisy)
        return noisy.clamp(0.0, 1.0)

    def build_condition_label(self, label, x_t, t, image):
        r = torch.rand(1, device=label.device).item()
        if r < self.gt_prob:
            return label
        if r < self.gt_prob + self.noisy_prob:
            return self._make_noisy_condition(label)
        if r < self.gt_prob + self.noisy_prob + self.self_prob:
            with torch.no_grad():
                first_pred = self.model(
                    x=x_t,
                    step=t,
                    image=image,
                    label=None,
                    pred_type="denoise",
                )
            return torch.sigmoid(first_pred).detach().clamp(0.0, 1.0)
        return None

    def get_input(self, batch):
        return batch["image"], batch["label"].float()

    def training_step(self, batch):
        image, label = self.get_input(batch)

        for p in self.model.parameters():
            p.grad = None

        with autocast():
            x_start = label * 2 - 1
            x_t, t, _ = self.model(x=x_start, pred_type="q_sample")
            cond_label = self.build_condition_label(label, x_t, t, image)
            pred_xstart = self.model(
                x=x_t,
                step=t,
                image=image,
                label=cond_label,
                pred_type="denoise",
            )

            loss_dice = self.dice_loss(pred_xstart, label)
            loss_bce = self.bce(pred_xstart, label)
            loss_mse = self.mse(torch.sigmoid(pred_xstart), label)
            loss = loss_dice + loss_bce + loss_mse

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log("train_loss", loss, step=self.global_step)
        self.log("train_loss_dice", loss_dice, step=self.global_step)
        self.log("train_loss_bce", loss_bce, step=self.global_step)
        self.log("train_loss_mse", loss_mse, step=self.global_step)

        if self.global_step % 20 == 0 and self.local_rank == 0:
            print(
                f"[Xin step {self.global_step} ep {self.epoch}] "
                f"loss={loss.item():.4f} "
                f"dice={loss_dice.item():.4f} "
                f"bce={loss_bce.item():.4f} "
                f"mse={loss_mse.item():.4f} | "
                f"cond GT={self.gt_prob:.2f} noisy={self.noisy_prob:.2f} "
                f"self={self.self_prob:.2f} none={self.none_prob:.2f}"
            )

        return loss

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
        target = label.cpu().numpy()
        wt = dice(output[:, 1], target[:, 1])
        tc = dice(output[:, 0], target[:, 0])
        et = dice(output[:, 2], target[:, 2])
        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs
        mean_dice = (wt + tc + et) / 3

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(self.model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model",
            )

        save_new_model_and_delete_last(
            self.model,
            os.path.join(self.model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model",
        )

        print(
            f"[Xin single-pass DDIM] wt={wt:.4f} tc={tc:.4f} "
            f"et={et:.4f} mean={mean_dice:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Single-stage fold-0 BraTS training")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env", default="pytorch")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--master-port", type=int, default=17840)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--infer-ddim-steps", type=int, default=10)
    parser.add_argument("--n-fold", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(args.seed)

    fold = 0
    fold_logdir = os.path.join(args.logdir, "fold0")

    train_dataset, val_dataset, test_dataset = get_loader_brats(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fold=fold,
        num_workers=args.num_workers,
        seed=args.seed,
        n_splits=args.n_fold,
    )

    trainer = XinTrainer(
        env_type=args.env,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        logdir=fold_logdir,
        val_every=args.val_every,
        num_gpus=args.num_gpus,
        master_port=args.master_port,
        training_script=__file__,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        infer_ddim_steps=args.infer_ddim_steps,
        num_workers=args.num_workers,
    )

    print("=" * 72)
    print("[Xin] Fold 0 only | single-stage training")
    print(f"  Dataset      : dataset/dataset.py")
    print(f"  Logdir       : {fold_logdir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Optimizer    : AdamW, lr={args.lr}, weight_decay=8e-4")
    print(f"  Scheduler    : warmup {args.warmup_epochs} epochs + cosine annealing")
    print("  Loss         : Dice + BCE + MSE = 1:1:1")
    print("  Condition    : GT=0.20, noisy=0.20, self-guided=0.40, none=0.20")
    print(f"  Inference    : single-pass DDIM, steps={args.infer_ddim_steps}")
    print("  Sliding win. : roi=96x96x96, overlap=0.5")
    print("  Two-pass eval: disabled")
    print("=" * 72)

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print(f"[Xin] Training finished. Held-out test dataset size: {len(test_dataset)}")


if __name__ == "__main__":
    main()


