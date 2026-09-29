"""Shared configuration, checkpoint, dataloader, and visualization helpers."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ImageFolder


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_dataloaders(data_root, resolution, batch_size, workers):
    train_set = ImageFolder(data_root, "train", resolution)
    test_set = ImageFolder(data_root, "test", resolution)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def checkpoint_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path).expanduser(), map_location=map_location)


def class_mask_rgb(class_mask):
    """Render cast as red and attached as green."""
    class_mask = class_mask.detach().cpu().numpy()
    rgb = np.zeros((*class_mask.shape[-2:], 3), dtype=np.uint8)
    rgb[..., 0] = (class_mask == 1) * 255
    rgb[..., 1] = (class_mask == 2) * 255
    return Image.fromarray(rgb)


def save_evaluation_panel(
    path,
    image,
    prediction,
    cast_mask,
    attached_mask,
    undefined_mask,
):
    image_array = (
        image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    ground_truth = torch.zeros_like(prediction)
    ground_truth[attached_mask.squeeze(0) > 0.5] = 2
    ground_truth[cast_mask.squeeze(0) > 0.5] = 1

    panels = [
        Image.fromarray(image_array),
        class_mask_rgb(ground_truth),
        class_mask_rgb(prediction),
        Image.fromarray(
            (
                (undefined_mask.squeeze(0) > 0.5)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
        ).convert("RGB"),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (len(panels) * width, height))
    for index, panel in enumerate(panels):
        canvas.paste(panel.convert("RGB"), (index * width, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
