"""Train the three-iteration cast/attached-shadow model."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data.augmentation import augment_batch, convert
from metrics import ShadowMetrics, checkpoint_score
from model.network import NetworkMultiClassNoMask
from utils import build_dataloaders, checkpoint_state, load_checkpoint, load_config


def light2attach(light, normal, tau=0.05):
    cosine = (normal * light.view(-1, 3, 1, 1)).sum(1, keepdim=True)
    return torch.sigmoid(cosine / tau)


def binary_dice_loss(logits, target, smooth=1e-6):
    prediction = torch.sigmoid(logits)
    intersection = (prediction * target).sum()
    cardinality = prediction.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)


def shadow_logits(segmentation_logits):
    background = segmentation_logits[:, 0]
    shadow = segmentation_logits[:, 1:3].logsumexp(dim=1)
    return shadow - background


def move_batch(batch, device):
    tensor_keys = [
        "image",
        "depth",
        "cast_mask",
        "attached_mask",
        "object_mask",
        "undefined_mask",
        "valid_mask",
        "light",
    ]
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def model_module(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def compute_loss(model, batch, epoch, config):
    image = batch["image"]
    depth = batch["depth"]
    cast = batch["cast_mask"] > 0.5
    attached = batch["attached_mask"] > 0.5
    undefined = batch["undefined_mask"] > 0.5
    valid_type = batch["valid_mask"] > 0.5

    full_target = (cast | attached | undefined).squeeze(1).float()
    class_target = torch.zeros_like(cast, dtype=torch.long)
    class_target[attached] = 2
    class_target[cast] = 1
    class_target = class_target.squeeze(1)
    valid_type = valid_type.squeeze(1)

    depth_valid = depth > 0
    normal = convert(depth) * depth_valid
    attached_prior = torch.ones_like(batch["attached_mask"])

    loss_config = config["loss"]
    class_weights = image.new_tensor([loss_config["background_weight"], 1.0, 1.0])
    positive_weight = image.new_tensor([loss_config["positive_weight"]])
    total_binary = image.new_zeros(())
    total_type = image.new_zeros(())
    total_light = image.new_zeros(())

    iterations = config["model"]["iterations"]
    for _ in range(iterations):
        segmentation, light = model(image, attached_prior, normal)
        pseudo_attached = light2attach(
            F.normalize(light, dim=1), normal, config["model"]["tau"]
        ) * depth_valid

        ground_truth_ratio = max(0.0, 1.0 - 0.1 * epoch)
        predicted_ratio = min(1.0, 0.1 * epoch)
        attached_prior = (
            predicted_ratio * pseudo_attached
            + ground_truth_ratio * attached.float()
        ).detach()

        binary_logits = shadow_logits(segmentation)
        total_binary += F.binary_cross_entropy_with_logits(
            binary_logits,
            full_target,
            pos_weight=positive_weight,
        )
        total_binary += loss_config["dice"] * binary_dice_loss(
            binary_logits, full_target
        )

        pixel_logits = segmentation.permute(0, 2, 3, 1)[valid_type]
        pixel_target = class_target[valid_type]
        total_type += F.cross_entropy(
            pixel_logits, pixel_target, weight=class_weights
        )

        difference = segmentation[:, 1] - segmentation[:, 2]
        cast_region = class_target == 1
        attached_region = class_target == 2
        cast_margin = (
            F.relu(0.2 - difference)[cast_region].sum()
            / cast_region.float().sum().clamp_min(1.0)
        )
        attached_margin = (
            F.relu(0.2 + difference)[attached_region].sum()
            / attached_region.float().sum().clamp_min(1.0)
        )
        total_type += loss_config["distinguish"] * (
            cast_margin + attached_margin
        )

        pseudo = pseudo_attached.squeeze(1)
        attached_target = attached.squeeze(1).float()
        confident_positive = attached.squeeze(1) & (pseudo > 0.5)
        pseudo_weight = (
            (valid_type & ~attached.squeeze(1)).float()
            + confident_positive.float()
            + 0.5 * (attached.squeeze(1) & ~confident_positive).float()
        )
        total_light += loss_config["attached_geometry"] * (
            F.binary_cross_entropy(
                pseudo,
                attached_target,
                weight=pseudo_weight,
                reduction="sum",
            )
            / pseudo_weight.sum().clamp_min(1.0)
        )
        total_light += loss_config["direction"] * F.l1_loss(
            light, batch["light"]
        )
        total_light += loss_config["unit"] * (
            (light.norm(dim=1) - 1.0) ** 2
        ).mean()

    total_binary /= iterations
    total_type /= iterations
    total_light /= iterations
    total = (
        loss_config["segmentation"] * total_binary
        + loss_config["type"] * total_type
        + loss_config["light"] * total_light
    )
    return total, {
        "segmentation": total_binary.detach(),
        "type": total_type.detach(),
        "light": total_light.detach(),
    }


@torch.no_grad()
def validate(model, loader, device, config):
    """Validate every epoch and return the historical checkpoint-selection score."""
    model.eval()
    metrics = ShadowMetrics(protocol="reported", attached_region="valid")
    iterations = config["model"]["iterations"]

    for batch in loader:
        batch = move_batch(batch, device)
        depth_valid = batch["depth"] > 0
        normal = convert(batch["depth"]) * depth_valid
        attached_prior = torch.ones_like(batch["attached_mask"])

        for _ in range(iterations):
            segmentation, light = model(batch["image"], attached_prior, normal)
            attached_prior = (
                light2attach(light, normal, config["model"]["tau"])
                * depth_valid
            ).detach()

        predicted_class = segmentation.argmax(dim=1)
        predicted_full = shadow_logits(segmentation) > 0
        metrics.update(
            predicted_full,
            predicted_class,
            (batch["cast_mask"] > 0.5).squeeze(1),
            (batch["attached_mask"] > 0.5).squeeze(1),
            (batch["object_mask"] > 0.5).squeeze(1),
            (batch["undefined_mask"] > 0.5).squeeze(1),
        )

    results = metrics.compute()
    score = checkpoint_score(results)
    for name in ("full", "cast", "attached"):
        values = results[name]
        print(
            f"{name.capitalize():8s} BER={values['ber']:.2f} "
            f"F1={values['f1']:.2f} P={values['precision']:.2f} "
            f"R={values['recall']:.2f}"
        )
    print(f"Checkpoint score: {score:.2f}")
    model.train()
    return score, results


def create_model(config, device):
    model = NetworkMultiClassNoMask(
        input_resolution=config["model"]["resolution"],
        light_in=8,
        pretrained_light=True,
    )
    initialization = load_checkpoint(
        config["paths"]["backbone_checkpoint"], map_location="cpu"
    )
    incompatible = model.load_state_dict(
        checkpoint_state(initialization), strict=False
    )
    print(
        "Loaded PVT initialization "
        f"({len(incompatible.missing_keys)} missing, "
        f"{len(incompatible.unexpected_keys)} unexpected keys)."
    )
    model.set_detector_input_channels(7)
    return model.to(device)


def main(config):
    torch.manual_seed(config["training"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(
        config["paths"]["data_root"],
        config["model"]["resolution"],
        config["training"]["batch_size"],
        config["training"]["workers"],
    )
    model = create_model(config, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    module = model_module(model)
    encoder_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name.startswith("encoder.") and "patch_embed1" not in name
    ]
    full_rate_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if not name.startswith("encoder.") or "patch_embed1" in name
    ]
    optimizer = torch.optim.Adamax(
        [
            {
                "params": encoder_parameters,
                "lr": config["training"]["learning_rate"] * 0.05,
            },
            {
                "params": full_rate_parameters,
                "lr": config["training"]["learning_rate"],
            },
        ]
    )

    output_dir = (
        Path(config["paths"]["output_dir"]) / config["experiment_name"]
    )
    checkpoint_dir = output_dir / "weights"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")

    best_score = float("-inf")
    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            if config["augmentation"]["enabled"]:
                batch = augment_batch(
                    batch,
                    config["augmentation"]["layers"],
                    config["augmentation"]["magnitude"],
                )
            batch = move_batch(batch, device)
            loss, parts = compute_loss(model, batch, epoch, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            writer.add_scalar("loss/total", loss.item(), global_step)
            for name, value in parts.items():
                writer.add_scalar(f"loss/{name}", value.item(), global_step)
            if batch_index % config["training"]["log_every"] == 0:
                print(
                    f"Epoch {epoch + 1:02d} batch {batch_index:04d}: "
                    f"loss={loss.item():.5f}"
                )

        score, _ = validate(model, test_loader, device, config)
        if score > best_score:
            best_score = score
            checkpoint_path = (
                checkpoint_dir / f"{score:.4f}_epoch{epoch + 1}.pth"
            )
            torch.save(
                {
                    "state_dict": model_module(model).state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "score": score,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data_root")
    parser.add_argument("--backbone_ckpt")
    parser.add_argument("--output_dir")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    if arguments.data_root:
        configuration["paths"]["data_root"] = arguments.data_root
    if arguments.backbone_ckpt:
        configuration["paths"]["backbone_checkpoint"] = arguments.backbone_ckpt
    if arguments.output_dir:
        configuration["paths"]["output_dir"] = arguments.output_dir
    main(configuration)
