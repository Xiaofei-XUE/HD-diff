"""
Dataset interface note for the public HD-Diff release.

The original project uses a private dataset-loading and preprocessing pipeline
for BraTS-style multimodal 3D MRI volumes. That implementation is not included
in this public release.

Expected data conceptually includes four MRI modalities per case, such as T1,
T1ce, T2, and FLAIR, together with tumor-region annotations used for supervised
segmentation experiments.

This placeholder is provided only to document where the dataset module belongs
in the full project structure. It intentionally does not contain file discovery,
medical image parsing, preprocessing transforms, data splitting, augmentation,
or training DataLoader construction.
"""
