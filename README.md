# HD-Diff: Hierarchical-Disentanglement-Guided Diffusion for Multimodal Brain Tumor Segmentation

This repository provides a public source release of HD-Diff, a hierarchical-disentanglement-guided diffusion framework for multimodal brain tumor segmentation. The released code includes the main model definition, core network modules, diffusion utilities, and non-executable placeholders for the dataset and training pipeline. Executable dataset loading, complete training/testing scripts, checkpoints, and experiment-specific runtime files are not included.

![Framework](imgs/figure2.jpg)

## Method Overview

HD-Diff processes four MRI modalities (T1, T1ce, T2, and FLAIR) with modality-aware encoder branches, then uses a dual-stream fusion module (DFM) to align cross-modal semantics and fuse multi-scale anatomical features.

The fused features are converted into structure-aware guidance by the boundary enhance conditioner (BEC) and core enhance conditioner (CEC). BEC guides shallow denoising stages for boundary refinement, while CEC guides deeper stages for tumor-core consistency. The diffusion-based denoising network then recovers the segmentation mask and predicts WT, TC, and ET subregions.

## Highlights

- Hierarchical-disentanglement-guided diffusion framework for multimodal brain tumor segmentation.
- Dual-stream fusion module (DFM) for cross-modal semantic alignment.
- Boundary enhance conditioner (BEC) and core enhance conditioner (CEC) for structure-aware denoising.
- Four-modality MRI input: T1, T1ce, T2, and FLAIR.
- Public model-side source code with dataset and training placeholders.

## Repository Structure

```text
.
+-- dataset/
|   +-- dataset.py
+-- guided_diffusion/
+-- light_training/
|   +-- evaluation/
|       +-- metric.py
+-- unet/
+-- hd_diff/
+-- model.py
+-- train.py
+-- demo_overview.py
+-- requirements.txt
+-- README.md
+-- imgs/
    +-- figure2.jpg
```

`model.py` exposes the main `DiffUNet` composition. The `unet/` and `guided_diffusion/` folders contain the corresponding network and diffusion components.

`dataset/dataset.py` is a text-only placeholder that documents where the private dataset loader belongs in the full project. It does not include file discovery, medical image reading, preprocessing, data splitting, augmentation, or DataLoader construction.

`train.py` is a non-runnable training-flow skeleton. It documents the expected training stages, but does not include executable optimization, validation, checkpointing, or experiment-launch logic.

## Environment

Install PyTorch according to your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

This project is designed around GPU-based 3D medical image segmentation. The complete training and evaluation pipeline is not part of this public release.

## Dataset Preparation

The full dataset loading and preprocessing implementation is not included. In the complete project, the dataset module is responsible for preparing BraTS-style multimodal MRI cases with four image modalities and tumor-region annotations.

A typical expected case contains:

```text
case_id/
+-- T1 image
+-- T1ce image
+-- T2 image
+-- FLAIR image
+-- segmentation label
```

Raw medical images are not included in this repository. Please download public datasets from their official sources and follow the corresponding licenses.

## Training

The public `train.py` file provides a high-level training-flow skeleton only. It is included to show the organization of the training process, but it intentionally stops before any private implementation is required.

In the complete experimental pipeline, training is organized as a single-stage fold-based procedure. The model first converts the ground-truth segmentation mask into the diffusion target space, samples a noisy mask through the forward diffusion process, and then predicts the denoised segmentation mask under image-conditioned guidance.

The training step follows this general pattern:

```text
1. read a batch of multimodal MRI volumes and tumor-region labels
2. map the label mask to the diffusion target representation
3. sample a diffusion timestep and generate a noisy mask
4. build a structure condition for the encoder
5. predict the denoised mask with the HD-Diff network
6. optimize the combined segmentation and denoising objective
7. log training losses and periodically validate the model
```

The structure condition used during training can come from several sources: ground-truth labels, perturbed/noisy labels, self-guided predictions, or an unconditioned branch. This mixed-condition design encourages the denoising network to remain useful when perfect structural guidance is unavailable.

The full implementation combines three losses with equal conceptual importance:

```text
loss = Dice loss + BCE loss + MSE loss
```

Validation is performed with sliding-window inference and a single-pass DDIM sampling procedure. The predicted channels are evaluated as WT, TC, and ET regions, and the mean Dice score is used for model selection in the private training pipeline.

The released `train.py` keeps this organization visible as a flow skeleton, but it does not include the executable dataset construction, optimizer and scheduler setup, mixed-precision runtime, validation loop, checkpoint saving/loading, launch configuration, or experiment-specific hyperparameters.

## Demo

A small overview demo is provided for printing the conceptual method flow:

```bash
python demo_overview.py
```

The demo is only a documentation helper. It does not train, test, or run medical image segmentation.

## Release Boundary

This public release focuses on the model architecture and method-level organization. The complete data-processing, training, and evaluation pipeline is not included because it depends on dataset licenses, dataset-specific preprocessing protocols, local compute infrastructure, and experiment-management settings. Users should prepare public datasets through their official sources and adapt the data interface according to the corresponding usage terms.

This public version intentionally does not include:

- Executable dataset loading or preprocessing code.
- Complete training or testing scripts.
- Checkpoint-loading and model-selection logic.
- Learned weights, logs, raw medical images, or generated predictions.
- Private experiment paths or local runtime artifacts.

## Citation

Citation will be updated after the manuscript is available.

## Acknowledgement

This project builds on ideas and components from MONAI, guided diffusion, UNet-based medical image segmentation, and the BraTS challenge ecosystem.



