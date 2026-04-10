import argparse
import json
import os
import resource
import sys
from datetime import datetime, timezone

import torch

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.loggers.default_logger import logger
from src.models.compose.composer import ModelComposer
from src.quantization.gdnsq.layers.exact_integer_conv2d import (
    BinarizedExactIntegerConv2d,
    ExactIntegerConv2d,
    derive_channel_thresholds_and_codes,
)
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator

torch.set_float32_matmul_precision("high")

_UNARY_TYPES = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Identity, torch.nn.Dropout)
_FLOAT_MODULE_TYPES = (
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    NoisyConv2d,
    NoisyLinear,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fused checkpoint into int-only oriented graph with hybrid binary fallback."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fused checkpoint after BatchNorm fusion (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="int_only.ckpt",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="int_only_report.json",
        help="Detailed export report (JSON).",
    )
    parser.add_argument(
        "--k-scan-radius",
        type=int,
        default=500_000,
        help="Scan radius for threshold search in derive_channel_thresholds_and_codes.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation before/after conversion.",
    )
    parser.add_argument(
        "--strict-int-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable strict policy: report float-oriented leftovers as an export policy violation (no crash).",
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


def build_quantized_model(config):
    model = ModelComposer(config=config).compose()
    quantizer = Quantizer(config=config)()
    qmodel = quantizer.quantize(model, in_place=True)
    return qmodel, quantizer


def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _find_following_unary_modules(parent: torch.nn.Module, child_name: str) -> list:
    unary = []
    keys = list(parent._modules.keys())
    try:
        idx = keys.index(child_name)
    except ValueError:
        return unary
    for key in keys[idx + 1 :]:
        m = parent._modules[key]
        if isinstance(m, _UNARY_TYPES):
            unary.append(m)
        else:
            break
    return unary


def _collect_exact_layers(root: torch.nn.Module):
    layers = []
    for name, module in root.named_modules():
        if not isinstance(module, ExactIntegerConv2d):
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = root.get_submodule(parent_name)
        else:
            parent = root
            child_name = name
        layers.append(
            {
                "name": name,
                "parent": parent,
                "child_name": child_name,
                "module": module,
            }
        )
    return layers


def _looks_like_residual_tail(layer_name: str) -> bool:
    # For the current ResNet-style model family, conv2 is followed by residual add.
    return ".conv2.conv" in layer_name


def _map_consumers_by_order(layers: list[dict]):
    consumer_map = {}
    for idx, entry in enumerate(layers):
        next_entry = layers[idx + 1] if idx + 1 < len(layers) else None
        consumer_map[entry["name"]] = next_entry
    return consumer_map


def _codes_to_consumer_float(codes: torch.Tensor, consumer: ExactIntegerConv2d) -> torch.Tensor:
    zp = consumer.azp * consumer.act_s / consumer.guard_a
    return zp + (codes / consumer.guard_a) * consumer.act_s


def _check_int_only_structure(root: torch.nn.Module):
    violations = []
    n_exact = 0
    n_binary = 0
    for name, module in root.named_modules():
        if isinstance(module, ExactIntegerConv2d):
            n_exact += 1
        if isinstance(module, BinarizedExactIntegerConv2d):
            n_binary += 1
        if isinstance(module, _FLOAT_MODULE_TYPES):
            violations.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                }
            )
    return {
        "n_exact_remaining": n_exact,
        "n_binary": n_binary,
        "float_module_violations": violations,
        "is_int_only_structural": len(violations) == 0,
    }


def export_hybrid_int_only(
    model: torch.nn.Module,
    k_scan_radius: int,
):
    root = _get_model_root(model)
    layers = _collect_exact_layers(root)
    consumers = _map_consumers_by_order(layers)

    report_layers = []
    n_binary = 0
    n_fallback = 0

    logger.info("Found %d ExactIntegerConv2d layers for export.", len(layers))
    for entry in layers:
        name = entry["name"]
        parent = entry["parent"]
        child_name = entry["child_name"]
        producer = entry["module"]
        unary_modules = _find_following_unary_modules(parent, child_name)
        consumer_entry = consumers[name]

        layer_report = {
            "name": name,
            "status": "integer_fallback",
            "reason": "",
            "unary_modules": [type(m).__name__ for m in unary_modules],
            "producer": {
                "act_s": float(producer.act_s.detach().cpu().item()),
                "azp": float(producer.azp.detach().cpu().item()),
                "guard_a": float(producer.guard_a.detach().cpu().item()),
            },
            "consumer": None,
            "thresholds": None,
        }

        if consumer_entry is None:
            layer_report["reason"] = "no_consumer_exact_layer"
            report_layers.append(layer_report)
            n_fallback += 1
            logger.info("Fallback %s: %s", name, layer_report["reason"])
            continue

        consumer = consumer_entry["module"]
        layer_report["consumer"] = {
            "name": consumer_entry["name"],
            "act_s": float(consumer.act_s.detach().cpu().item()),
            "azp": float(consumer.azp.detach().cpu().item()),
            "guard_a": float(consumer.guard_a.detach().cpu().item()),
        }

        if _looks_like_residual_tail(name):
            layer_report["reason"] = "residual_tail_requires_joint_derive"
            report_layers.append(layer_report)
            n_fallback += 1
            logger.info("Fallback %s: %s", name, layer_report["reason"])
            continue

        try:
            th, code_lo, code_hi = derive_channel_thresholds_and_codes(
                producer.fp_scale,
                producer.bias,
                consumer,
                unary_modules,
                k_scan_radius,
            )
            code_lo_f = _codes_to_consumer_float(code_lo, consumer)
            code_hi_f = _codes_to_consumer_float(code_hi, consumer)

            new_layer = BinarizedExactIntegerConv2d(producer)
            new_layer.th.copy_(th)
            new_layer.code_lo.copy_(code_lo_f)
            new_layer.code_hi.copy_(code_hi_f)
            parent._modules[child_name] = new_layer

            layer_report["status"] = "binary"
            layer_report["reason"] = "ok"
            layer_report["thresholds"] = {
                "th_min": int(th.min().item()),
                "th_max": int(th.max().item()),
                "code_lo_min": float(code_lo_f.min().item()),
                "code_lo_max": float(code_lo_f.max().item()),
                "code_hi_min": float(code_hi_f.min().item()),
                "code_hi_max": float(code_hi_f.max().item()),
            }
            n_binary += 1
            logger.info("Binary %s: th=[%d, %d]", name, int(th.min().item()), int(th.max().item()))
        except Exception as exc:
            layer_report["reason"] = f"derive_failed: {exc}"
            n_fallback += 1
            logger.warning("Fallback %s: %s", name, layer_report["reason"])

        report_layers.append(layer_report)

    return {
        "layers": report_layers,
        "summary": {
            "total_exact_layers": len(layers),
            "binary_layers": n_binary,
            "integer_fallback_layers": n_fallback,
        },
    }


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

    logger.info("Validating before export")
    before_metrics = _validate_if_enabled(
        validator,
        qmodel,
        data,
        checkpoint=args.checkpoint,
        enabled=args.validate,
    )
    logger.info("Validation before export: %s", before_metrics)

    # load from fused checkpoint into already fused structure to preserve previous pipeline compatibility
    logger.info("Loading fused checkpoint into model for conversion: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = qmodel.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys while loading checkpoint: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %d", len(unexpected))

    logger.info("Exporting int-only hybrid graph with k_scan_radius=%d", args.k_scan_radius)
    export_report = export_hybrid_int_only(
        qmodel,
        args.k_scan_radius,
    )

    structural = _check_int_only_structure(_get_model_root(qmodel))
    logger.info(
        "Structural check: binary=%d, exact=%d, float_violations=%d",
        structural["n_binary"],
        structural["n_exact_remaining"],
        len(structural["float_module_violations"]),
    )
    strict_policy_violation = args.strict_int_only and not structural["is_int_only_structural"]
    if strict_policy_violation:
        n_violations = len(structural["float_module_violations"])
        logger.error("strict-int-only policy violation: found %d float-oriented module(s)", n_violations)
        for v in structural["float_module_violations"][:20]:
            logger.error("  float module: %s (%s)", v["name"], v["type"])
        if n_violations > 20:
            logger.error("  ... and %d more", n_violations - 20)

    logger.info("Validating after export")
    after_metrics = _validate_if_enabled(
        validator,
        qmodel,
        data,
        checkpoint=None,
        enabled=args.validate,
    )
    logger.info("Validation after export: %s", after_metrics)

    validator.save_checkpoint(filepath=args.output)
    logger.info("Saved int-only checkpoint to %s", args.output)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "k_scan_radius": args.k_scan_radius,
            "validate": args.validate,
            "strict_int_only": args.strict_int_only,
        },
        "metrics": {
            "before": _to_python(before_metrics),
            "after": _to_python(after_metrics),
        },
        "conversion": export_report,
        "structural_check": structural,
        "status": {
            "strict_int_only_policy_violation": strict_policy_violation,
        },
        "output": {
            "checkpoint": args.output,
            "report": args.report,
        },
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(_to_python(report), f, indent=2, ensure_ascii=True)
    logger.info("Saved detailed report to %s", args.report)


if __name__ == "__main__":
    main()
