"""Training augmentations and depth-to-normal conversion."""

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from torchvision import transforms


def convert(depth):
    """Convert Bx1xHxW relative depth maps to unit surface normals."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape Bx1xHxW")
    gradient = spatial_gradient(depth, normalized=False).squeeze(1)
    z = torch.ones_like(depth)
    normal = torch.cat((-gradient, z), dim=1)
    return F.normalize(normal, dim=1, eps=1e-8)


def _distraction(image, magnitude):
    """Darken one random convex region without altering annotations."""
    _, _, height, width = image.shape
    span = max(2, int(min(height, width) * magnitude * 1.6))
    random_index = random.randint(1, 120)
    edge_count = 3 + random_index % 5
    x0 = random.randint(0, max(0, width - span))
    y0 = random.randint(0, max(0, height - span))
    points = []
    for _ in range(edge_count):
        points.append([
            random.randint(x0, min(width - 1, x0 + span - 1)),
            random.randint(y0, min(height - 1, y0 + span - 1)),
        ])

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.asarray(points), 1.0)
    mask = torch.from_numpy(mask)[None, None].to(
        device=image.device, dtype=image.dtype
    )
    kernel = torch.ones((1, 1, 3, 3), device=image.device, dtype=image.dtype)
    mask = F.conv2d(mask, kernel, padding=1) > 5
    attenuation = (~mask).to(image.dtype) + 0.2 * mask.to(image.dtype)
    return image * attenuation


def augment_batch(batch, layers=1, magnitude=0.3):
    """Apply the augmentations used by the released training configuration."""
    operations = random.choices(
        ["blur", "brightness", "sharpness", "contrast", "distraction", "flip"],
        k=layers,
    )
    for operation in operations:
        if operation == "blur":
            kernel = int(13 * magnitude)
            if kernel % 2 != 1:
                kernel += 1
            batch["image"] = transforms.GaussianBlur(
                kernel, 2 * magnitude
            )(batch["image"])
        elif operation == "brightness":
            batch["image"] = transforms.ColorJitter(
                brightness=0.9 * magnitude
            )(batch["image"])
        elif operation == "sharpness":
            batch["image"] = transforms.RandomAdjustSharpness(
                10 * magnitude, p=1
            )(batch["image"])
        elif operation == "contrast":
            batch["image"] = transforms.ColorJitter(
                contrast=0.7 * magnitude
            )(batch["image"])
        elif operation == "distraction":
            batch["image"] = _distraction(batch["image"], magnitude)
        else:
            flip = transforms.RandomHorizontalFlip(p=1)
            for key in (
                "image",
                "cast_mask",
                "attached_mask",
                "object_mask",
                "undefined_mask",
                "depth",
            ):
                batch[key] = flip(batch[key])
            batch["valid_mask"] = 1.0 - batch["undefined_mask"]
            batch["light"][:, 0] *= -1
    return batch

