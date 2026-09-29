"""Test the released checkpoint on the official 292-image test split."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.augmentation import convert
from data.dataset import ImageFolder
from metrics import ShadowMetrics
from model.network import NetworkMultiClassNoMask
from utils import checkpoint_state, load_checkpoint, load_config, save_evaluation_panel


def light2attach(light, normal, tau=0.05):
    cosine = (normal * light.view(-1, 3, 1, 1)).sum(1, keepdim=True)
    return torch.sigmoid(cosine / tau).detach()


def shadow_logits(segmentation_logits):
    background = segmentation_logits[:, 0]
    shadow = segmentation_logits[:, 1:3].logsumexp(dim=1)
    return shadow - background


def run_iterations(model, image, normal, depth_valid, iterations, tau):
    attached_prior = torch.ones_like(depth_valid)
    segmentation = light = None
    for _ in range(iterations):
        segmentation, light = model(image, attached_prior, normal)
        attached_prior = light2attach(light, normal, tau) * depth_valid
    return segmentation, light


def tiled_prediction(model, image, normal, depth_valid, config):
    """Run the four local crops and global image used by the paper evaluation."""
    crop_size = config["crop_size"]
    resolution = image.shape[-1]
    tile_size = resolution // crop_size
    images, normals, valid_masks = [], [], []

    for row in range(crop_size):
        for column in range(crop_size):
            ys = slice(row * tile_size, (row + 1) * tile_size)
            xs = slice(column * tile_size, (column + 1) * tile_size)
            images.append(
                F.interpolate(
                    image[:, :, ys, xs],
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            normals.append(
                F.interpolate(
                    normal[:, :, ys, xs],
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            valid_masks.append(
                F.interpolate(
                    depth_valid[:, :, ys, xs],
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            )

    image_batch = torch.cat(images + [image], dim=0)
    normal_batch = torch.cat(normals + [normal], dim=0)
    valid_batch = torch.cat(valid_masks + [depth_valid], dim=0)
    logits, lights = run_iterations(
        model,
        image_batch,
        normal_batch,
        valid_batch,
        config["iterations"],
        config["tau"],
    )

    global_logits = logits[-1:]
    local_logits = torch.zeros_like(global_logits)
    for row in range(crop_size):
        for column in range(crop_size):
            index = row * crop_size + column
            tile = F.interpolate(
                logits[index : index + 1],
                size=(tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            )
            ys = slice(row * tile_size, (row + 1) * tile_size)
            xs = slice(column * tile_size, (column + 1) * tile_size)
            local_logits[:, :, ys, xs] = tile

    global_probability = torch.softmax(global_logits, dim=1)
    local_probability = torch.softmax(local_logits, dim=1)
    use_local = (
        (global_probability > config["filter_threshold"])
        & (local_probability > global_probability)
    )
    fused_probability = torch.where(
        use_local, local_probability, global_probability
    )
    predicted_class = fused_probability.argmax(dim=1)
    predicted_full = (
        torch.logsumexp(fused_probability[:, 1:3], dim=1)
        - fused_probability[:, 0]
        > 0
    )
    return predicted_class, predicted_full, lights[-1:]


def print_metrics(results):
    for name in ("full", "cast", "attached"):
        values = results[name]
        print(f"\n{name.upper()}")
        print(f"BER:       {values['ber']:.4f}")
        print(f"S:         {values['shadow_error']:.4f}")
        print(f"NS:        {values['nonshadow_error']:.4f}")
        print(f"Precision: {values['precision']:.4f}")
        print(f"Recall:    {values['recall']:.4f}")
        print(f"F1:        {values['f1']:.4f}")


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = config["evaluation"]["protocol"]
    if protocol not in {"standard", "reported"}:
        raise ValueError("evaluation.protocol must be standard or reported")
    print(f"Evaluation protocol: {protocol}")
    dataset = ImageFolder(
        config["paths"]["data_root"],
        split="test",
        resolution=config["evaluation"]["resolution"],
        evaluation_protocol=protocol,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["evaluation"]["workers"],
        pin_memory=True,
    )

    model = NetworkMultiClassNoMask(
        input_resolution=config["evaluation"]["resolution"],
        light_in=8,
        pretrained_light=False,
    )
    model.set_detector_input_channels(7)
    checkpoint = load_checkpoint(
        config["paths"]["model_checkpoint"], map_location="cpu"
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.to(device).eval()

    metrics = ShadowMetrics(protocol=protocol)
    light_errors = []
    elapsed = 0.0
    output_dir = Path(config["paths"]["output_dir"])
    if config["evaluation"]["save_visuals"]:
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            depth_valid = (depth > 0).float()
            normal = convert(depth) * depth_valid

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            if config["evaluation"]["crop"]:
                predicted_class, predicted_full, light = tiled_prediction(
                    model,
                    image,
                    normal,
                    depth_valid,
                    {
                        **config["evaluation"],
                        "iterations": config["model"]["iterations"],
                        "tau": config["model"]["tau"],
                    },
                )
            else:
                logits, light = run_iterations(
                    model,
                    image,
                    normal,
                    depth_valid,
                    config["model"]["iterations"],
                    config["model"]["tau"],
                )
                predicted_class = logits.argmax(dim=1)
                predicted_full = shadow_logits(logits) > 0

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start

            cast = batch["cast_mask"].to(device).squeeze(1)
            attached = batch["attached_mask"].to(device).squeeze(1)
            object_mask = batch["object_mask"].to(device).squeeze(1)
            undefined = batch["undefined_mask"].to(device).squeeze(1)
            metrics.update(
                predicted_full,
                predicted_class,
                cast,
                attached,
                object_mask,
                undefined,
            )
            target_light = batch["light"].to(device)
            light_errors.append((light - target_light).abs().cpu())

            if config["evaluation"]["save_visuals"]:
                save_evaluation_panel(
                    output_dir / f"{index:04d}_{batch['sample_id'][0]}.png",
                    image[0],
                    predicted_class[0],
                    batch["cast_mask"][0],
                    batch["attached_mask"][0],
                    batch["undefined_mask"][0],
                )
            if (index + 1) % 10 == 0:
                print(
                    f"Evaluated {index + 1}/{len(loader)} "
                    f"(average inference {elapsed / (index + 1):.5f}s)"
                )

    results = metrics.compute()
    print_metrics(results)
    light_error = torch.cat(light_errors).mean(dim=0)
    print(
        "\nLight absolute error: "
        f"x={light_error[0]:.4f}, y={light_error[1]:.4f}, "
        f"z={light_error[2]:.4f}"
    )
    print(f"Average inference time: {elapsed / len(loader):.5f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--data_root")
    parser.add_argument("--ckpt")
    parser.add_argument("--save_dir")
    parser.add_argument(
        "--protocol",
        choices=("standard", "reported"),
        help=(
            "Evaluation protocol; overrides evaluation.protocol in the config."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print all metrics without creating visualization files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    if arguments.data_root:
        configuration["paths"]["data_root"] = arguments.data_root
    if arguments.metrics_only:
        configuration["evaluation"]["save_visuals"] = False
    if arguments.ckpt:
        configuration["paths"]["model_checkpoint"] = arguments.ckpt
    if arguments.save_dir:
        configuration["paths"]["output_dir"] = arguments.save_dir
    if arguments.protocol:
        configuration["evaluation"]["protocol"] = arguments.protocol
    main(configuration)
