# HD-Diff: Hierarchical-Disentanglement-Guided Diffusion for Multimodal Brain Tumor Segmentation

This repository provides the implementation of HD-Diff, a hierarchical-disentanglement-guided diffusion framework for multimodal brain tumor segmentation. The model uses four MRI modalities as input and integrates a modality-aware encoder, a dual-stream fusion module (DFM), boundary/core enhance conditioners, and a diffusion-based denoising network to predict three tumor subregions.

![Framework](imgs/figure2.jpg)

## Method Overview

HD-Diff processes four BraTS2020 MRI modalities (T1, T1ce, T2, and FLAIR) with modality-specific encoder branches, then uses a dual-stream fusion module (DFM) to align cross-modal semantics and fuse multi-scale anatomical features.

The fused features are converted into structure-aware guidance by the boundary enhance conditioner (BEC) and core enhance conditioner (CEC). BEC guides shallow denoising stages for boundary refinement, while CEC guides deeper stages for tumor-core consistency. The diffusion-based denoising network then recovers the segmentation mask and predicts WT, TC, and ET subregions with sliding-window inference for full 3D volumes.

## Highlights

- Hierarchical-disentanglement-guided diffusion framework for multimodal brain tumor segmentation.
- Dual-stream fusion module (DFM) for cross-modal semantic alignment.
- Boundary enhance conditioner (BEC) and core enhance conditioner (CEC) for structure-aware denoising.
- BraTS2020 four-modality input: T1, T1ce, T2, and FLAIR.
- Sliding-window inference with Dice and HD95 evaluation.

## Repository Structure

```text
.
+-- dataset/
+-- guided_diffusion/
+-- light_training/
+-- unet/
+-- train_xin.py
+-- requirements.txt
+-- README.md
```

## Environment

Create a Python environment:

```bash
conda create -n diffunet-brats python=3.8 -y
conda activate diffunet-brats
```

Install PyTorch according to your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

This project is designed for GPU training. Please choose the PyTorch and CUDA versions that match your machine.

## Dataset Preparation

The dataset loader is implemented in `dataset/dataset.py`. The public example below uses the BraTS2020-style folder structure.

Example BraTS2020-style structure:

```text
data/
+-- MICCAI_BraTS2020_TrainingData/
    +-- BraTS20_Training_001/
    |   +-- BraTS20_Training_001_t1.nii.gz
    |   +-- BraTS20_Training_001_t1ce.nii.gz
    |   +-- BraTS20_Training_001_t2.nii.gz
    |   +-- BraTS20_Training_001_flair.nii.gz
    |   +-- BraTS20_Training_001_seg.nii.gz
    +-- ...
```


Raw BraTS data is not included in this repository. Please download the dataset from the official challenge source and follow its license.

## Training

Run single-stage training with `train_xin.py`:

```bash
python train_xin.py \
  --data-dir ./data/MICCAI_BraTS2020_TrainingData \
  --logdir ./logs_xin \
  --device cuda:0 \
  --batch-size 8 \
  --epochs 300
```

Common options:

```text
--data-dir          Path to BraTS data
--logdir            Directory for logs and checkpoints
--device            Training device, for example cuda:0
--batch-size        Training batch size
--num-workers       Number of data loading workers
--val-every         Validation interval
--epochs            Number of training epochs
--lr                Learning rate
--warmup-epochs     Warmup epochs for the scheduler
--infer-ddim-steps  DDIM sampling steps during validation
--n-fold            Number of folds for data splitting
```

The default input patch size is:

```text
ROI_SIZE = [96, 96, 96]
```

Checkpoints and training logs are saved under:

```text
logs_xin/
logs_xin/fold0/model/
```

Do not commit raw medical images, large checkpoints, logs, or generated prediction results to GitHub.


## Citation

Citation will be updated after the manuscript is available.

## Acknowledgement

This project builds on ideas and components from MONAI, guided diffusion, UNet-based medical image segmentation, and the BraTS challenge ecosystem.

