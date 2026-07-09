# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K?
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset

BRATS2024_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
BRATS2020_MODALITIES = ("t1", "t1ce", "t2", "flair")


def _resolve_case_dir(data_path):
    normalized = os.path.normpath(data_path)
    if normalized.endswith(".nii.gz") or os.path.isfile(normalized):
        return os.path.dirname(normalized)
    return normalized


def _build_case_files(case_dir):
    case_name = os.path.basename(os.path.normpath(case_dir))

    brats2024_images = [
        os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        for modality in BRATS2024_MODALITIES
    ]
    brats2024_seg = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
    if all(os.path.exists(path) for path in brats2024_images) and os.path.exists(brats2024_seg):
        return brats2024_images, brats2024_seg

    file_identifizer = case_name.split("_")[-1]
    brats2020_images = [
        os.path.join(case_dir, f"BraTS20_Training_{file_identifizer}_{modality}.nii.gz")
        for modality in BRATS2020_MODALITIES
    ]
    brats2020_seg = os.path.join(case_dir, f"BraTS20_Training_{file_identifizer}_seg.nii.gz")
    if all(os.path.exists(path) for path in brats2020_images) and os.path.exists(brats2020_seg):
        return brats2020_images, brats2020_seg

    return None


def _find_case_dirs(data_dir):
    root = _resolve_case_dir(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    if _build_case_files(root) is not None:
        return [root]

    case_dirs = []
    for name in os.listdir(root):
        case_dir = os.path.join(root, name)
        if os.path.isdir(case_dir) and _build_case_files(case_dir) is not None:
            case_dirs.append(case_dir)

    if not case_dirs:
        raise FileNotFoundError(f"No BraTS case folders were found under: {data_dir}")
    return case_dirs


def _split_train_val_test(all_paths, fold=0, n_splits=5, seed=1234):
    all_paths = list(all_paths)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_paths)

    size = len(all_paths)
    if size == 1:
        return all_paths, all_paths, all_paths

    test_size = max(1, int(round(size * 0.2)))
    if test_size >= size:
        test_size = 1

    test_files = all_paths[:test_size]
    train_val_files = all_paths[test_size:]

    if len(train_val_files) <= 1:
        return train_val_files, train_val_files, test_files

    actual_splits = min(n_splits, len(train_val_files))
    kfold = KFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    fold_items = list(kfold.split(np.arange(len(train_val_files))))
    train_idx, val_idx = fold_items[fold % actual_splits]

    train_files = [train_val_files[i] for i in train_idx]
    val_files = [train_val_files[i] for i in val_idx]
    return train_files, val_files, test_files


def resample_img(
    image: sitk.Image,
    out_spacing=(2.0, 2.0, 2.0),
    out_size=None,
    is_label: bool = False,
    pad_value=0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):
        case_dir = _resolve_case_dir(data_path)
        case_files = _build_case_files(case_dir)
        if case_files is None:
            raise FileNotFoundError(f"Missing BraTS files in case folder: {case_dir}")

        image_paths, seg_path = case_files

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = np.array(image_data).astype(np.float32)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
        return {
            "image": image_data,
            "label": seg_data
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"{self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfoldKFolf?
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res


class Args:
    def __init__(self) -> None:
        self.workers = 8
        self.fold = 0
        self.batch_size = 2


def get_loader_brats(data_dir, batch_size=1, fold=0, num_workers=8, seed=1234, n_splits=5):

    all_paths = _find_case_dirs(data_dir)
    train_files, val_files, test_files = _split_train_val_test(
        all_paths,
        fold=fold,
        n_splits=n_splits,
        seed=seed,
    )

    print(
        f"fold is {fold}, train is {len(train_files)}, "
        f"val is {len(val_files)}, test is {len(test_files)}"
    )

    rotate_rad = math.radians(15.0)
    train_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
                                        random_size=False),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.75, spatial_axes=(0, 1)),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.75, spatial_axes=(1, 2)),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.75, spatial_axes=(0, 2)),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.30,
                rotate_range=(rotate_rad, rotate_rad, rotate_rad),
                scale_range=(0.10, 0.10, 0.10),
                padding_mode="border",
                spatial_size=(96, 96, 96),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.RandGaussianNoised(keys="image", prob=0.20, mean=0.0, std=0.05),
            transforms.RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.15,
            ),
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = PretrainDataset(train_files, transform=train_transform)

    val_ds = PretrainDataset(val_files, transform=val_transform)

    test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_ds, test_ds]

    return loader
