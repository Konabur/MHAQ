import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.loggers.default_logger import logger
from src.models.compose.composer import ModelComposer
from src.quantization.gdnsq.layers.exact_integer_conv2d import ExactIntegerConv2d
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PTQ for identity shortcuts: quantize 1x1 conv shortcuts to reduce float modules."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (YAML).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fused checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fused_ptq.ckpt",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="fused_ptq_report.json",
        help="Detailed export report (JSON).",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=10,
        help="Number of batches for calibration.",
    )
    parser.add_argument(
        "--weight-bit",
        type=int,
        default=8,
        help="Weight quantization bits (default: 8).",
    )
    parser.add_argument(
        "--act-bit",
        type=int,
        default=8,
        help="Activation quantization bits (default: 8).",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation before/after conversion.",
    )
    return parser.parse_args()


def _to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def _get_model_root(model: nn.Module) -> nn.Module:
    return model.model if hasattr(model, "model") else model


def identify_identity_shortcuts(root: nn.Module) -> list[dict]:
    """Find identity shortcut conv+bn pairs."""
    shortcuts = []
    for name, module in root.named_modules():
        if "identity_conv" in name and isinstance(module, nn.Conv2d):
            # Find parent to get bn
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                parent = root.get_submodule(parent_name)
                if hasattr(parent, "bn") and isinstance(parent.bn, nn.BatchNorm2d):
                    shortcuts.append({
                        "name": name,
                        "conv": module,
                        "bn": parent.bn,
                        "parent": parent,
                    })
    return shortcuts


def find_corresponding_conv2(root: nn.Module, identity_name: str) -> ExactIntegerConv2d | None:
    """Find the conv2 layer that shares residual add with this identity shortcut."""
    # identity_name like: features.stage2.unit1.identity_conv.conv
    # corresponding conv2: features.stage2.unit1.body.conv2.conv
    parts = identity_name.split(".")
    if "identity_conv" in parts:
        idx = parts.index("identity_conv")
        conv2_name = ".".join(parts[:idx] + ["body", "conv2", "conv"])
        try:
            conv2 = root.get_submodule(conv2_name)
            if isinstance(conv2, ExactIntegerConv2d):
                return conv2
        except AttributeError:
            pass
    return None


class ActivationCalibrator:
    """Collect activation statistics for calibration."""

    def __init__(self):
        self.stats = {}
        self.hooks = []

    def register_hook(self, module: nn.Module, name: str):
        def hook(mod, inp, out):
            if name not in self.stats:
                self.stats[name] = {"min": [], "max": []}
            self.stats[name]["min"].append(out.detach().min().cpu().item())
            self.stats[name]["max"].append(out.detach().max().cpu().item())

        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_scales(self, name: str, bits: int) -> tuple[float, float]:
        """Compute act_s and azp from collected statistics."""
        if name not in self.stats:
            raise ValueError(f"No statistics collected for {name}")

        min_val = min(self.stats[name]["min"])
        max_val = max(self.stats[name]["max"])

        # Symmetric quantization around zero for simplicity
        # act_s = scale, azp = zero_point
        n_levels = 2 ** bits
        scale = (max_val - min_val) / (n_levels - 1)
        azp = 0.0  # symmetric

        return scale, azp


def convert_float_conv_bn_to_exact_integer(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    target_act_s: float,
    weight_bit: int = 8,
    act_bit: int = 8,
) -> ExactIntegerConv2d:
    """Convert float Conv2d + BN to ExactIntegerConv2d with PTQ."""

    # Fuse BN into conv
    with torch.no_grad():
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        std = torch.sqrt(var + bn.eps)
        bn_scale = gamma / std
        bn_shift = beta - mean * bn_scale

    # Create adapter that mimics NoisyConv2d interface
    class PTQConvAdapter(nn.Module):
        def __init__(self, conv, target_act_s, weight_bit, act_bit):
            super().__init__()
            self.weight = conv.weight
            self.bias = conv.bias if conv.bias is not None else None
            self.stride = conv.stride
            self.padding = conv.padding
            self.dilation = conv.dilation
            self.groups = conv.groups
            self.padding_mode = conv.padding_mode
            self.kernel_size = conv.kernel_size
            self.out_channels = conv.out_channels
            self.in_channels = conv.in_channels
            self.weight_bit = weight_bit
            self.act_bit = act_bit

            # Quantization parameters
            self.act_guard_bit = 0  # guard_a = 1
            self.weight_guard_bit = 0  # guard_w = 1

            # Use log2 for consistency with NoisyConv2d
            self.log_act_s = nn.Parameter(torch.log2(torch.tensor(target_act_s, device=conv.weight.device)))
            act_levels = 2 ** act_bit
            self.log_act_q = nn.Parameter(torch.log2(torch.tensor(float(act_levels), device=conv.weight.device)))
            self.act_b = nn.Parameter(torch.tensor(0.0, device=conv.weight.device))  # azp = 0

        def _configure_weight_quantizer(self):
            # Per-channel symmetric quantization
            w = self.weight
            # Compute per-channel scale: max absolute value
            w_max = w.abs().amax(dim=(1, 2, 3), keepdim=False)
            n_levels = 2 ** self.weight_bit
            weight_scale = w_max / (n_levels / 2 - 1)  # symmetric: [-127, 127] for 8-bit
            weight_zero_point = torch.zeros_like(weight_scale)
            return weight_scale, weight_zero_point

    # Create adapter instance
    adapter = PTQConvAdapter(conv, target_act_s, weight_bit, act_bit)

    # Add Q attribute with quantize method
    class Quantizer:
        def __init__(self, weight_bit):
            self.weight_bit = weight_bit

        def quantize(self, weight):
            # Symmetric quantization
            w_max = weight.abs().amax(dim=(1, 2, 3), keepdim=True)
            n_levels = 2 ** self.weight_bit
            scale = w_max / (n_levels / 2 - 1)
            scale = torch.clamp(scale, min=1e-8)  # avoid division by zero
            w_quant = torch.clamp(torch.round(weight / scale), min=-(n_levels // 2), max=(n_levels // 2 - 1))
            return w_quant

    adapter.Q = Quantizer(weight_bit)

    # Create ExactIntegerConv2d using adapter
    exact_conv = ExactIntegerConv2d(
        adapter,
        in_scale=torch.tensor(target_act_s, device=conv.weight.device),
        post_scale=bn_scale,
        post_shift=bn_shift,
    )

    return exact_conv


def _validate_if_enabled(validator, model, data, checkpoint=None, enabled=True):
    if not enabled:
        return None
    if checkpoint is None:
        return validator.validate(model, datamodule=data)
    return validator.validate(model, datamodule=data, ckpt_path=checkpoint)


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    data = DatasetComposer(config=config).compose()
    qmodel, quantizer = build_quantized_model(config)
    validator = Validator(config=config, logger=False, callbacks=False)

    logger.info("Fusing BatchNorm to match fused checkpoint structure")
    quantizer.fuse_conv_bn(qmodel)

    logger.info("Loading fused checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = qmodel.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info("Missing keys: %d", len(missing))
    if unexpected:
        logger.info("Unexpected keys: %d", len(unexpected))

    logger.info("Validating before PTQ")
    before_metrics = _validate_if_enabled(
        validator,
        qmodel,
        data,
        checkpoint=None,
        enabled=args.validate,
    )
    logger.info("Validation before PTQ: %s", before_metrics)

    root = _get_model_root(qmodel)
    shortcuts = identify_identity_shortcuts(root)
    logger.info("Found %d identity shortcuts", len(shortcuts))

    if len(shortcuts) == 0:
        logger.error("No identity shortcuts found - nothing to quantize")
        return

    # Calibration phase
    logger.info("Starting calibration with %d batches", args.calibration_batches)
    calibrator = ActivationCalibrator()

    for shortcut in shortcuts:
        calibrator.register_hook(shortcut["conv"], shortcut["name"])

    qmodel.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data.val_dataloader()):
            if batch_idx >= args.calibration_batches:
                break
            x, y = batch
            qmodel(x)

    calibrator.remove_hooks()
    logger.info("Calibration complete")

    # Convert identity shortcuts
    conversion_report = []
    for shortcut in shortcuts:
        name = shortcut["name"]
        conv = shortcut["conv"]
        bn = shortcut["bn"]
        parent = shortcut["parent"]

        # Find corresponding conv2 to match quantization domain
        conv2 = find_corresponding_conv2(root, name)
        if conv2 is None:
            logger.warning("Could not find corresponding conv2 for %s - using calibrated scale", name)
            act_s, azp = calibrator.compute_scales(name, bits=1)
        else:
            # Match conv2's activation scale
            act_s = conv2.act_s.item()
            azp = conv2.azp.item()
            logger.info("Matching %s to conv2 act_s=%.6f", name, act_s)

        # Convert to ExactIntegerConv2d
        try:
            exact_conv = convert_float_conv_bn_to_exact_integer(
                conv, bn, target_act_s=act_s, weight_bit=args.weight_bit, act_bit=args.act_bit
            )

            # Replace in model
            parent.conv = exact_conv
            parent.bn = nn.Identity()  # Remove BN

            conversion_report.append({
                "name": name,
                "status": "quantized",
                "act_s": act_s,
                "azp": azp,
            })
            logger.info("Converted %s to ExactIntegerConv2d", name)
        except Exception as exc:
            conversion_report.append({
                "name": name,
                "status": "failed",
                "error": str(exc),
            })
            logger.error("Failed to convert %s: %s", name, exc)

    # Structural check
    float_modules = []
    for name, module in root.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            if not isinstance(module, ExactIntegerConv2d):
                float_modules.append({"name": name, "type": type(module).__name__})

    logger.info("Structural check: %d float modules remaining", len(float_modules))

    logger.info("Validating after PTQ")
    after_metrics = _validate_if_enabled(
        validator,
        qmodel,
        data,
        checkpoint=None,
        enabled=args.validate,
    )
    logger.info("Validation after PTQ: %s", after_metrics)

    # Check accuracy delta
    if before_metrics and after_metrics:
        before_top1 = before_metrics[0]["Accuracy_top1"]
        after_top1 = after_metrics[0]["Accuracy_top1"]
        delta = after_top1 - before_top1
        logger.info("Accuracy delta: %.4f (%.2f%%)", delta, delta * 100)

        if abs(delta) > 0.001:  # 0.1%
            logger.warning("Accuracy delta exceeds 0.1%% threshold")

    validator.save_checkpoint(filepath=args.output)
    logger.info("Saved PTQ checkpoint to %s", args.output)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "calibration_batches": args.calibration_batches,
            "validate": args.validate,
        },
        "metrics": {
            "before": _to_python(before_metrics),
            "after": _to_python(after_metrics),
        },
        "conversion": {
            "identity_shortcuts_found": len(shortcuts),
            "identity_shortcuts_quantized": sum(1 for r in conversion_report if r["status"] == "quantized"),
            "details": conversion_report,
        },
        "structural_check": {
            "float_modules_remaining": len(float_modules),
            "float_modules": float_modules[:20],  # limit output
        },
        "output": {
            "checkpoint": args.output,
            "report": args.report,
        },
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(_to_python(report), f, indent=2, ensure_ascii=True)
    logger.info("Saved detailed report to %s", args.report)


def build_quantized_model(config):
    model = ModelComposer(config=config).compose()
    quantizer = Quantizer(config=config)()
    qmodel = quantizer.quantize(model, in_place=True)
    return qmodel, quantizer


if __name__ == "__main__":
    main()
