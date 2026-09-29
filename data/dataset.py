"""Dataset loader for the ECCV cast/attached-shadow dataset."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ImageFolder(Dataset):
    """Load an official train or test split.

    undefined_mask == 1 marks pixels whose shadow type is undefined.
    valid_mask is provided explicitly as its complement. Both evaluation
    protocols resize masks bilinearly; the evaluator determines how fractional
    boundary values are handled.
    """

    SPLIT_FILES = {
        "train": "Three_train_data.csv",
        "test": "Three_test_data.csv",
        "validate": "Three_test_data.csv",
    }

    MASK_INTERPOLATION = {
        "reported": InterpolationMode.BILINEAR,
        "standard": InterpolationMode.BILINEAR,
    }

    def __init__(
        self,
        root,
        split="train",
        resolution=512,
        evaluation_protocol="reported",
    ):
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unknown split {split!r}; choose train or test")
        if evaluation_protocol not in self.MASK_INTERPOLATION:
            raise ValueError(
                "evaluation_protocol must be standard or reported"
            )
        self.root = Path(root).expanduser().resolve()
        self.split = "test" if split == "validate" else split
        self.resolution = int(resolution)
        self.mask_interpolation = self.MASK_INTERPOLATION[
            evaluation_protocol
        ]
        csv_path = self.root / self.SPLIT_FILES[split]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        self.samples = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.samples)

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_mask(self, value):
        mask = Image.open(self._path(value)).convert("L")
        mask = TF.resize(
            mask,
            [self.resolution, self.resolution],
            interpolation=self.mask_interpolation,
        )
        return TF.pil_to_tensor(mask).float() / 255.0

    def _convert_light(self, raw_light, original_size):
        """Convert a stored raw light tuple to the unit-vector training target."""
        x, y, raw_z = (float(v) for v in ast.literal_eval(str(raw_light)))
        width, height = original_size
        x *= self.resolution / width
        y *= self.resolution / height
        z = raw_z / 255.0 if raw_z != -0.5 else -0.5

        xy_norm = float(np.hypot(x, y))
        target_xy_norm = float(np.sqrt(max(0.0, 1.0 - z * z)))
        if xy_norm > 0:
            x *= target_xy_norm / xy_norm
            y *= target_xy_norm / xy_norm
        else:
            x = y = 0.0
        return torch.tensor([x, y, z], dtype=torch.float32)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        image_path = self._path(row["img"])
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = TF.resize(
            image,
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.BILINEAR,
        )

        depth = np.load(self._path(row["depth"])).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        depth = TF.resize(
            torch.from_numpy(depth),
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.BILINEAR,
        )

        undefined_mask = self._load_mask(row["undefined"])
        return {
            "image": TF.pil_to_tensor(image).float() / 255.0,
            "depth": depth,
            "cast_mask": self._load_mask(row["cast"]),
            "attached_mask": self._load_mask(row["attached"]),
            "object_mask": self._load_mask(row["object"]),
            "undefined_mask": undefined_mask,
            "valid_mask": 1.0 - undefined_mask,
            "light": self._convert_light(row["light"], original_size),
            "image_path": str(image_path),
            "sample_id": image_path.stem,
        }


