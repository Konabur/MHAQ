"""
Fuse BatchNorm into the previous NoisyConv2d, normalize activation scales,
run validation, print weight/bias value-set stats, and save fused checkpoint.
"""
import os
import sys
import resource

# Disable experiment tracing/logging (e.g., WandbLogger) for this script.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_START_METHOD", "thread")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.quantization.gdnsq.layers.gdnsq_act_lin import NoisyActLin
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.training.trainer import Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision("high")

def _disable_tracing(config):
    # Prevent construction of WandbLogger inside src/training/trainer.py
    try:
        tloggers = config.training.loggers
        if isinstance(tloggers, dict):
            tloggers.pop("WandbLogger", None)
            tloggers.pop("wandb", None)
            tloggers.pop("wandb_logger", None)
    except Exception:
        pass


def _config_for_fuse_validation(config):
    """
    Fused models drop NoisyConv2d / NoisyActLin; training callbacks that call
    model_stats.is_converged() must not run. Use a config copy with no callbacks.
    """
    t = config.training.model_copy(update={"callbacks": {}})
    return config.model_copy(update={"training": t})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse BatchNorm, normalize activation scales, validate, print stats, save fused checkpoint."
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
        help="Path to the trained quantized checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the fused checkpoint.",
    )
    return parser.parse_args()


class _ExactIntegerConv2d(torch.nn.Module):
    """
    Exact integer-input replacement for NoisyConv2d with inlined activation quant.

    The activation integer zero-point offset is folded using a **position-independent**
    correction: for constant code-domain input ``azp``, each output channel of
    ``conv(azp * 1, guard_w * w + wzp)`` equals ``azp * sum(guard_w * w + wzp)`` over
    that channel's kernel (ignoring padding / border effects). That yields a 1×C×1×1
    bias instead of a full-resolution map (which would track image size).
    """

    def __init__(
        self,
        module: NoisyConv2d,
        in_scale: torch.Tensor,
        post_scale: torch.Tensor | None = None,
        post_shift: torch.Tensor | None = None,
    ):
        super().__init__()
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.kernel_size = module.kernel_size
        # Compute weight codes and quant params exactly like NoisyActLin.forward.
        weight_scale, weight_zero_point = module._configure_weight_quantizer()
        w = module.Q.quantize(module.weight)
        p_scale = (
            post_scale.reshape(-1)
            if post_scale is not None
            else torch.ones(module.out_channels, device=w.device, dtype=w.dtype)
        )
        p_shift = (
            post_shift.reshape(-1)
            if post_shift is not None
            else torch.zeros(module.out_channels, device=w.device, dtype=w.dtype)
        )
        if module.bias is None:
            b = torch.zeros(module.out_channels, device=w.device, dtype=w.dtype)
        else:
            b = module.bias

        self.act_guard_bit = module.act_guard_bit
        self.weight_guard_bit = module.weight_guard_bit

        # x-independent precomputations for forward()
        guard_a = float(2 ** int(self.act_guard_bit))
        guard_w = float(2 ** int(self.weight_guard_bit))

        # Integer activation zero-point in code-domain (azp = round(act_b / s * guard_a))
        azp = torch.round(module.act_b / torch.exp2(module.log_act_s) * guard_a).reshape([])

        # Weight integer zero-point in code-domain (wzp = round(guard_w * weight_zero_point)).
        # Note: weight_zero_point comes from NoisyActLin._configure_weight_quantizer()
        # as qwmin / guard_w (code-domain), so multiplying by guard_w yields integer qwmin.
        wzp = torch.round(guard_w * weight_zero_point).reshape(-1)

        # Precompute constant FP scale.
        fp_scale = (
            in_scale
            * weight_scale.reshape(1, -1, 1, 1)
            * p_scale.view(1, -1, 1, 1)
            / (guard_a * guard_w)
        )
        bias_base = b.view(1, -1, 1, 1) / p_scale.view(1, -1, 1, 1) + p_shift.view(1, -1, 1, 1)

        wzp_kernel = wzp.view(-1, 1, 1, 1).expand_as(w)
        wk = guard_w * w + wzp_kernel
        d_per_out_ch = azp * wk.sum(dim=(1, 2, 3))
        bias_spatial = (d_per_out_ch.view(1, -1, 1, 1) * fp_scale)
        bias_full = bias_base + bias_spatial

        # Register buffers at the end.
        self.register_buffer("weight", w.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer(
            "act_s",
            torch.exp2(module.log_act_s).reshape([]).detach().clone().to(device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer(
            "act_q",
            torch.exp2(module.log_act_q).reshape([]).detach().clone().to(device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer("azp", azp.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("wzp", wzp.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("fp_scale", fp_scale.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer("bias", bias_full.detach().clone().to(device=w.device, dtype=w.dtype), persistent=True)
        self.register_buffer(
            "guard_a",
            torch.tensor(guard_a, device=w.device, dtype=w.dtype),
            persistent=True,
        )
        self.register_buffer(
            "guard_w",
            torch.tensor(guard_w, device=w.device, dtype=w.dtype),
            persistent=True,
        )

    def _conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            pad_h, pad_w = self.padding
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, None, self.stride, padding, self.dilation, self.groups)

    def forward(self, x):
        # Activation quantize
        zp = self.azp * self.act_s / self.guard_a
        x = torch.clamp(x, min=zp, max=zp + self.act_q - self.act_s)
        x = self.guard_a * torch.round((x - zp) / self.act_s)

        # Integer-valued convolution
        c1 = self._conv(x, self.weight)
        c2 = self._conv(x, torch.ones_like(self.weight))
        out = self.guard_w * c1 + self.wzp.view(1, -1, 1, 1) * c2

        bias = self.bias.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return out * self.fp_scale + bias


def _iter_quantized_affines(root: torch.nn.Module):
    """Yield fused GDNSQ affine layers in module order."""
    for name, module in root.named_modules():
        if isinstance(module, (NoisyConv2d, NoisyLinear)):
            parts = name.split(".")
            child_name = parts[-1]
            parent_name = ".".join(parts[:-1]) if len(parts) > 1 else ""
            parent = root.get_submodule(parent_name) if parent_name else root
            yield name, parent, child_name, module, module


def _get_quantized_weight_params(
    module: NoisyConv2d | NoisyLinear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "_get_quantized_weight_params was inlined into _ExactIntegerConv2d.__init__."
    )


def _make_exact_integer_affine_from_quantized(
    module: NoisyConv2d | NoisyLinear,
    input_scale: torch.Tensor,
    post_scale: torch.Tensor | None = None,
    post_shift: torch.Tensor | None = None,
) -> torch.nn.Module:
    if isinstance(module, NoisyConv2d):
        return _ExactIntegerConv2d(module, input_scale, post_scale, post_shift)
    raise ValueError(f"Unsupported module type: {type(module)}")

def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def fuse_batchnorm_and_normalize_activation_scales(model: torch.nn.Module) -> int:
    """
    Walk the model once to:
      1) Fuse BatchNorm layers into the preceding NoisyConv2d and remove the BatchNorm from the graph.
      2) Replace each fused activation-aware affine layer with an inline integer-output activation
         and an exact integer affine that absorbs both the activation dequantization and any fused
         BatchNorm post-affine parameters.
    Returns (n_fused_batchnorm, n_normalized_activations).
    """
    root = _get_model_root(model)
    batch_norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )
    fused_count = 0

    # Materialize the list of quantized affine sites first to avoid mutating
    # the module hierarchy (removing BatchNorm, replacing children) while
    # PyTorch is still traversing it via named_modules.
    affine_sites = list(_iter_quantized_affines(root))

    # Single logical pass over quantized affine layers:
    # for each one, optionally fuse a following BatchNorm and immediately
    # rewrite the node into an exact integer affine module with inlined activation.
    for name, parent, child_name, act, qmodule in affine_sites:
        ref = qmodule.weight  # device/dtype reference
        current_scale = torch.exp2(act.log_act_s).to(device=ref.device, dtype=ref.dtype)
        post_scale = None
        post_shift = None

        # If this affine is a NoisyConv2d and is immediately followed by a BatchNorm,
        # fold that BatchNorm into per-channel post_scale/post_shift and remove it.
        if isinstance(qmodule, NoisyConv2d):
            keys = list(parent._modules.keys())
            try:
                idx = keys.index(child_name)
            except ValueError:
                idx = -1

            if 0 <= idx < len(keys) - 1:
                next_key = keys[idx + 1]
                next_module = parent._modules[next_key]
                if isinstance(next_module, batch_norm_types):
                    batch_norm = next_module
                    with torch.no_grad():
                        mean = batch_norm.running_mean
                        var = batch_norm.running_var
                        gamma = batch_norm.weight
                        beta = batch_norm.bias
                        std = torch.sqrt(var + batch_norm.eps)
                        bn_scale = gamma / std
                        bn_post_shift = beta - mean * bn_scale
                        post_scale = bn_scale.to(device=ref.device, dtype=ref.dtype)
                        post_shift = bn_post_shift.to(device=ref.device, dtype=ref.dtype)

                    # Remove the BatchNorm node from the execution graph.
                    # For Sequential containers we can drop it entirely; for
                    # other parents (e.g., ConvBlock with a .bn attribute) we
                    # must leave a placeholder Module so attribute access in
                    # forward() still works.
                    if isinstance(parent, torch.nn.Sequential):
                        parent._modules.pop(next_key, None)
                    else:
                        setattr(parent, next_key, torch.nn.Identity())
                    fused_count += 1
                    logger.info(
                        "Removed BatchNorm and folded into %s: post_scale/post_shift.",
                        name,
                    )

        with torch.no_grad():
            exact_affine = _make_exact_integer_affine_from_quantized(
                qmodule,
                current_scale,
                post_scale,
                post_shift,
            )
            parent._modules[child_name] = exact_affine

        # Count of normalized activations is no longer returned, but we keep
        # this transformation for side effects and detailed logging.
        logger.info(
            "Replaced fused activation in %s with inline round-clamp act and exact integer affine: act_scale=%s.",
            name,
            current_scale.cpu().item() if current_scale.numel() == 1 else current_scale.flatten()[0].cpu().item(),
        )

    return fused_count


def print_weight_bias_stats(model: torch.nn.Module):
    """Print per-layer weight value-sets for integer affine layers."""
    root = _get_model_root(model)
    for name, mod in root.named_modules():
        if isinstance(mod, _ExactIntegerConv2d):
            wvals = sorted(torch.unique(mod.weight.detach()).cpu().tolist())
            logger.info(
                "layer=%s weight_values=%s",
                name,
                wvals,
            )


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

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    qmodel.load_state_dict(state, strict=False)
    qmodel.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        qmodel = qmodel.to("cuda")

    n_fused_batchnorm = fuse_batchnorm_and_normalize_activation_scales(qmodel)
    logger.info(
        "Fused %d BatchNorm layer(s) into previous NoisyConv2d and removed from graph.",
        n_fused_batchnorm,
    )
    # Remove GDNSQ validation decoration so transformed inference-only models
    # can be validated without quantizer-width bookkeeping.
    qmodel.validation_step = type(qmodel).validation_step.__get__(qmodel, type(qmodel))

    root = _get_model_root(qmodel)
    logger.info("Model (after fusion):\n%s", root)

    logger.info("Running validation on fused model")
    validator.validate(qmodel, datamodule=data)

    logger.info("Weight and bias value-set stats (unique counts):")
    print_weight_bias_stats(qmodel)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": qmodel.state_dict()}, args.output)
    logger.info("Saved fused checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
