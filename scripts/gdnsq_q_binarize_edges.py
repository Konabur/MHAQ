"""
Replace each interior ExactIntegerConv2d with BinarizedExactIntegerConv2d.

Takes the fused checkpoint produced by ``gdnsq_q_fuse_batchnorm.py`` (same
``--config`` YAML) and, for each producer ``ExactIntegerConv2d`` whose output
feeds directly into another ``ExactIntegerConv2d``, derives per-channel integer
thresholds and binary codes via ``derive_channel_thresholds_and_codes`` and
swaps in a ``BinarizedExactIntegerConv2d``.  The final conv in the network
(no integer consumer) is intentionally left as ``ExactIntegerConv2d``.
"""
import os
import sys
import resource

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_START_METHOD", "thread")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.quantization.gdnsq.layers.exact_integer_conv2d import (
    ExactIntegerConv2d,
    BinarizedExactIntegerConv2d,
    derive_channel_thresholds_and_codes,
)
from src.training.trainer import Validator
from src.loggers.default_logger import logger

# Re-use helpers from the fuse script (no circular dependency — pure functions).
from scripts.gdnsq_q_fuse_batchnorm import (
    _disable_tracing,
    _config_for_fuse_validation,
    _get_model_root,
    materialize_exact_integer_convs_no_batchnorm_fuse,
)

torch.set_float32_matmul_precision("high")

# Module types whose presence between two ExactIntegerConv2d layers is safe to
# include in the unary chain passed to derive_channel_thresholds_and_codes.
_UNARY_TYPES = (
    torch.nn.ReLU,
    torch.nn.LeakyReLU,
    torch.nn.Identity,
    torch.nn.Dropout,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace interior ExactIntegerConv2d with BinarizedExactIntegerConv2d, "
            "validate, and save binarized checkpoint."
        )
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config (same as used for fuse_batchnorm).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Fused checkpoint produced by gdnsq_q_fuse_batchnorm.py.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the binarized checkpoint.")
    return parser.parse_args()


def _binarize_container(container: torch.nn.Module) -> int:
    """
    Scan direct children of *container* for (ExactIntegerConv2d, [unary...],
    ExactIntegerConv2d) triplets and replace the producer with
    BinarizedExactIntegerConv2d.  Then recurse into sub-containers.

    Returns the number of layers binarized.
    """
    n_binarized = 0
    # Snapshot keys so we can safely mutate _modules while iterating.
    children_snapshot = list(container._modules.items())

    producer_key = None
    producer = None
    unary_chain: list[torch.nn.Module] = []

    for key, mod in children_snapshot:
        if isinstance(mod, ExactIntegerConv2d):
            if producer is not None:
                # Found a consumer — binarize the producer.
                consumer = mod
                try:
                    with torch.no_grad():
                        th, code_lo, code_hi = derive_channel_thresholds_and_codes(
                            producer.fp_scale, producer.bias, consumer, unary_chain
                        )
                        binarized = BinarizedExactIntegerConv2d(producer)
                        binarized.th.copy_(th)
                        binarized.code_lo.copy_(code_lo)
                        binarized.code_hi.copy_(code_hi)
                    container._modules[producer_key] = binarized
                    n_binarized += 1
                    logger.info(
                        "Binarized %s → th=[%d, %d] codes_unique=%s",
                        producer_key,
                        int(th.min().item()),
                        int(th.max().item()),
                        sorted(torch.unique(
                            torch.cat([code_lo.flatten(), code_hi.flatten()])
                        ).tolist()),
                    )
                except ValueError as exc:
                    logger.warning(
                        "Skipping binarization of %s: %s", producer_key, exc
                    )
            # The current mod becomes the new producer candidate.
            producer_key = key
            producer = mod
            unary_chain = []

        elif isinstance(mod, _UNARY_TYPES):
            if producer is not None:
                unary_chain.append(mod)
            # else: unary op before any producer — ignore.

        else:
            # Non-unary, non-conv module breaks the producer chain.
            producer = None
            producer_key = None
            unary_chain = []

    # Recurse into sub-containers (skip leaf quantized layers).
    for _key, mod in children_snapshot:
        if isinstance(mod, (ExactIntegerConv2d, BinarizedExactIntegerConv2d)):
            continue
        if hasattr(mod, "_modules") and mod._modules:
            n_binarized += _binarize_container(mod)

    return n_binarized


def binarize_edges(model: torch.nn.Module) -> int:
    """Walk the full model and binarize all interior ExactIntegerConv2d layers."""
    root = _get_model_root(model)
    return _binarize_container(root)


def print_binarized_stats(model: torch.nn.Module):
    root = _get_model_root(model)
    for name, mod in root.named_modules():
        if isinstance(mod, BinarizedExactIntegerConv2d):
            logger.info(
                "layer=%s  channels=%d  th_range=[%d, %d]  "
                "code_lo_unique=%s  code_hi_unique=%s",
                name,
                mod.weight.shape[0],
                int(mod.th.min().item()),
                int(mod.th.max().item()),
                sorted(torch.unique(mod.code_lo).tolist()),
                sorted(torch.unique(mod.code_hi).tolist()),
            )


def _count_modules(model: torch.nn.Module, cls):
    return sum(1 for _, m in _get_model_root(model).named_modules() if isinstance(m, cls))


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    _disable_tracing(config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=_config_for_fuse_validation(config))

    data = dataset_composer.compose()
    model = model_composer.compose()
    qmodel = quantizer.quantize(model, in_place=True)

    # Materialise ExactIntegerConv2d skeletons so the fused state_dict loads cleanly.
    n_materialized = materialize_exact_integer_convs_no_batchnorm_fuse(qmodel)
    logger.info("Materialized %d ExactIntegerConv2d layer(s) for checkpoint loading.", n_materialized)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    qmodel.load_state_dict(state, strict=False)
    qmodel.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        qmodel = qmodel.to("cuda")

    n_total = _count_modules(qmodel, ExactIntegerConv2d)
    logger.info("ExactIntegerConv2d layers before binarization: %d", n_total)

    n_binarized = binarize_edges(qmodel)

    n_remaining = _count_modules(qmodel, ExactIntegerConv2d)
    logger.info(
        "Binarized %d/%d layer(s); %d left as ExactIntegerConv2d (tail/skipped).",
        n_binarized, n_total, n_remaining,
    )

    logger.info("Model (after binarization):\n%s", _get_model_root(qmodel))
    print_binarized_stats(qmodel)

    # Remove GDNSQ validation hooks so inference-only model validates cleanly.
    qmodel.validation_step = type(qmodel).validation_step.__get__(qmodel, type(qmodel))

    logger.info("Running validation on binarized model.")
    validator.validate(qmodel, datamodule=data)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": qmodel.state_dict()}, args.output)
    logger.info("Saved binarized checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
