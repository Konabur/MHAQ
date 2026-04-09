import os
import sys
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator
from src.loggers.default_logger import logger
from src.quantization.gdnsq.layers.exact_integer_conv2d import (
    ExactIntegerConv2d,
    BinarizedExactIntegerConv2d,
    derive_channel_thresholds_and_codes,
)

torch.set_float32_matmul_precision("high")

_UNARY_TYPES = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Identity, torch.nn.Dropout)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert fused ExactIntegerConv2d model to fully binarized BinarizedExactIntegerConv2d"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fused checkpoint after BatchNorm fusion (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="binarized.ckpt",
        help="Output binarized checkpoint path.",
    )
    parser.add_argument(
        "--k-scan-radius",
        type=int,
        default=500_000,
        help="Scan radius for threshold search in derive_channel_thresholds_and_codes.",
    )
    return parser.parse_args()


def build_quantized_model(config):
    model = ModelComposer(config=config).compose()
    quantizer = Quantizer(config=config)()
    qmodel = quantizer.quantize(model, in_place=True)
    return qmodel, quantizer


def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def find_following_unary_modules(parent: torch.nn.Module, child_name: str) -> list:
    """Collect unary modules (ReLU, Identity, Dropout, ...) that follow child_name in parent's module order."""
    unary = []
    keys = list(parent._modules.keys())
    try:
        idx = keys.index(child_name)
    except ValueError:
        return unary
    for key in keys[idx + 1:]:
        m = parent._modules[key]
        if isinstance(m, _UNARY_TYPES):
            unary.append(m)
        else:
            break
    return unary


def binarize_exact_integer_convs(model: torch.nn.Module, k_scan_radius: int) -> int:
    root = _get_model_root(model)

    # Collect all ExactIntegerConv2d layers with parent info
    layers = []
    for name, module in root.named_modules():
        if isinstance(module, ExactIntegerConv2d):
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = root
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            else:
                parent = root
                child_name = name
            layers.append((name, parent, child_name, module))

    logger.info("Found %d ExactIntegerConv2d layers to binarize", len(layers))

    n_binarized = 0
    for name, parent, child_name, exact_conv in layers:
        try:
            # After fusion, activation is already inlined in ExactIntegerConv2d's fp_scale/bias
            # Do NOT pass any unary modules - they would be applied twice
            unary_modules = []
            logger.info("Binarizing %s (activation already fused)", name)

            th, code_lo, code_hi = derive_channel_thresholds_and_codes(
                exact_conv.fp_scale,
                exact_conv.bias,
                exact_conv,
                unary_modules,
                k_scan_radius,
            )

            new_layer = BinarizedExactIntegerConv2d(exact_conv)
            new_layer.th.copy_(th)
            new_layer.code_lo.copy_(code_lo)
            new_layer.code_hi.copy_(code_hi)

            parent._modules[child_name] = new_layer
            n_binarized += 1
            logger.info(
                "Binarized %s: th_range=[%d, %d]",
                name,
                int(th.min().item()),
                int(th.max().item()),
            )
        except Exception as e:
            logger.warning("Failed to binarize layer %s: %s", name, e)

    return n_binarized


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    data = DatasetComposer(config=config).compose()
    qmodel, quantizer = build_quantized_model(config)
    validator = Validator(config=config, logger=False, callbacks=False)

    logger.info("Fusing BatchNorm to match fused checkpoint structure")
    quantizer.fuse_conv_bn(qmodel)

    logger.info("Loading fused checkpoint: %s", args.checkpoint)
    logger.info("Validating before binarization")
    before_metrics = validator.validate(
        qmodel,
        datamodule=data,
        ckpt_path=args.checkpoint,
    )
    logger.info("Validation before binarization: %s", before_metrics)

    logger.info("Performing binarization with k_scan_radius=%d", args.k_scan_radius)
    n_binarized = binarize_exact_integer_convs(qmodel, args.k_scan_radius)

    if n_binarized == 0:
        logger.warning("No ExactIntegerConv2d layers were binarized. Is this a fused checkpoint?")
        return

    logger.info("Binarized %d ExactIntegerConv2d layer(s).", n_binarized)

    logger.info("Validating after binarization")
    after_metrics = validator.validate(qmodel, datamodule=data)
    logger.info("Validation after binarization: %s", after_metrics)

    validator.save_checkpoint(filepath=args.output)
    logger.info("Saved binarized checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
