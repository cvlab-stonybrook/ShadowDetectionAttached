"""Pixel-level metrics for standard comparison and paper reproduction."""

import torch


class ShadowMetrics:
    """Accumulate full, cast, and attached-shadow metrics.

    The standard protocol thresholds masks at 0.5 and uses conventional binary
    confusion matrices. The reported protocol keeps fractional resized boundary
    pixels and reproduces the evaluation used for the paper's reported results.
    """

    def __init__(self, protocol="standard", attached_region="object"):
        if protocol not in {"standard", "reported"}:
            raise ValueError("protocol must be standard or reported")
        if attached_region not in {"object", "valid"}:
            raise ValueError("attached_region must be object or valid")
        self.protocol = protocol
        self.attached_region = attached_region
        self.counts = {
            name: {key: 0 for key in ("tp", "fp", "fn", "tn")}
            for name in ("full", "cast", "attached")
        }
        self.valid_total = 0
        self.attached_region_total = 0

    def _update(self, name, prediction, target, region):
        prediction = prediction.bool()
        positive = target == 1
        negative = target == 0
        region = region.bool()
        counts = self.counts[name]
        counts["tp"] += int((region & prediction & positive).sum())
        counts["fp"] += int((region & prediction & negative).sum())
        counts["fn"] += int((region & ~prediction & positive).sum())
        counts["tn"] += int((region & ~prediction & negative).sum())

    def update(
        self,
        pred_full,
        pred_class,
        cast_mask,
        attached_mask,
        object_mask,
        undefined_mask,
    ):
        if self.protocol == "standard":
            cast_mask = cast_mask > 0.5
            attached_mask = attached_mask > 0.5
            object_mask = object_mask > 0.5
            undefined_mask = undefined_mask > 0.5

        valid = (
            undefined_mask == 0
            if self.protocol == "reported"
            else ~undefined_mask
        )
        object_region = (
            object_mask == 1
            if self.protocol == "reported"
            else object_mask
        )
        attached_region = (
            object_region if self.attached_region == "object" else valid
        )
        full_target = torch.clamp(
            undefined_mask.float()
            + cast_mask.float()
            + attached_mask.float(),
            min=0,
            max=1,
        )

        self._update(
            "full", pred_full, full_target, torch.ones_like(pred_full)
        )
        self._update("cast", pred_class == 1, cast_mask, valid)
        self._update(
            "attached", pred_class == 2, attached_mask, attached_region
        )
        self.valid_total += int(valid.sum())
        self.attached_region_total += int(attached_region.sum())

    @staticmethod
    def _compute(counts, negative_total=None):
        tp, fp = counts["tp"], counts["fp"]
        fn, tn = counts["fn"], counts["tn"]
        positive = tp + fn
        negative = tn + fp if negative_total is None else negative_total
        precision = 100.0 * tp / max(tp + fp, 1)
        recall = 100.0 * tp / max(positive, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        shadow_error = 100.0 * fn / max(positive, 1)
        nonshadow_error = 100.0 * fp / max(negative, 1)
        return {
            "ber": 0.5 * (shadow_error + nonshadow_error),
            "shadow_error": shadow_error,
            "nonshadow_error": nonshadow_error,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def compute(self):
        if self.protocol == "standard":
            return {
                name: self._compute(self.counts[name])
                for name in ("full", "cast", "attached")
            }

        cast_positive = self.counts["cast"]["tp"] + self.counts["cast"]["fn"]
        attached_positive = (
            self.counts["attached"]["tp"] + self.counts["attached"]["fn"]
        )
        return {
            "full": self._compute(self.counts["full"]),
            "cast": self._compute(
                self.counts["cast"], self.valid_total - cast_positive
            ),
            "attached": self._compute(
                self.counts["attached"],
                self.attached_region_total - attached_positive,
            ),
        }


def checkpoint_score(results):
    """Score historically used by validate() to select the best checkpoint."""
    return (
        0.6 * results["full"]["f1"]
        + 0.2 * results["cast"]["f1"]
        + 0.2 * results["attached"]["f1"]
    )
