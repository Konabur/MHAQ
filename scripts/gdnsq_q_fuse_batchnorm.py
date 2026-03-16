"""
Fuse BatchNorm into the previous NoisyConv2d, normalize activation scales,
run validation, print weight/bias value-set stats, and save fused checkpoint.
"""
import os
import sys
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse
import torch.nn.functional as F
from collections import OrderedDict

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


def _to_scalar_buffer(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Ensure value is a buffer on the same device/dtype as like (0-dim if single element)."""
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), device=like.device, dtype=like.dtype)
    t = value.detach().clone().to(device=like.device, dtype=like.dtype)
    return t.reshape([]) if t.numel() == 1 else t.flatten()[0].reshape([])


class _ExactIntegerConv2d(torch.nn.Module):
    """
    Exact integer-input replacement for NoisyConv2d, with optional inlined
    activation quantization parameters so that activation and affine are a
    single module.
    """

    def __init__(
        self,
        module: NoisyConv2d,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor | float,
        input_zero_point: torch.Tensor | float,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        post_scale: torch.Tensor | None = None,
        post_shift: torch.Tensor | None = None,
        activation_module: NoisyActLin | None = None,
    ):
        super().__init__()
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.kernel_size = module.kernel_size
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        self.register_buffer("bias", bias.detach().clone(), persistent=True)
        self.register_buffer(
            "input_scale",
            _to_scalar_buffer(input_scale, weight),
            persistent=True,
        )
        self.register_buffer(
            "input_zero_point",
            _to_scalar_buffer(input_zero_point, weight),
            persistent=True,
        )
        self.register_buffer("weight_scale", weight_scale.detach().clone(), persistent=True)
        self.register_buffer(
            "weight_zero_point",
            weight_zero_point.detach().clone().to(device=weight.device, dtype=weight.dtype),
            persistent=True,
        )
        out_channels = weight.shape[0]
        default_post_scale = torch.ones(out_channels, device=weight.device, dtype=weight.dtype)
        default_post_shift = torch.zeros(out_channels, device=weight.device, dtype=weight.dtype)
        self.register_buffer(
            "post_scale",
            (
                post_scale.detach().clone().to(device=weight.device, dtype=weight.dtype).reshape(-1)
                if post_scale is not None
                else default_post_scale
            ),
            persistent=True,
        )
        self.register_buffer(
            "post_shift",
            (
                post_shift.detach().clone().to(device=weight.device, dtype=weight.dtype).reshape(-1)
                if post_shift is not None
                else default_post_shift
            ),
            persistent=True,
        )

        # Optional inlined activation quantizer parameters (from NoisyActLin).
        if activation_module is not None:
            self.register_buffer(
                "log_act_s",
                activation_module.log_act_s.detach().clone().to(device=weight.device, dtype=weight.dtype),
                persistent=True,
            )
            self.register_buffer(
                "log_act_q",
                activation_module.log_act_q.detach().clone().to(device=weight.device, dtype=weight.dtype),
                persistent=True,
            )
            self.register_buffer(
                "act_b",
                activation_module.act_b.detach().clone().to(device=weight.device, dtype=weight.dtype),
                persistent=True,
            )
            self.disable_activation = bool(getattr(activation_module, "disable", False))
        else:
            self.log_act_s = None
            self.log_act_q = None
            self.act_b = None
            self.disable_activation = True

    def _conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            pad_h, pad_w = self.padding
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, None, self.stride, padding, self.dilation, self.groups)

    def _conv1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolution with an all-ones kernel matching self.weight's shape,
        implemented via avg_pool2d + channel reduction for efficiency.
        """
        n, c, _, _ = x.shape
        k_h, k_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        # Avg pooling per channel gives spatial mean over the kernel window
        pooled = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=self.stride, padding=self.padding)
        # Convert to sum over spatial window
        patch_area = k_h * k_w
        summed_spatial = pooled * patch_area  # shape: (N, C, H_out, W_out)

        # Sum over channels within each group to match grouped conv with ones kernel
        groups = self.groups
        in_per_group = c // groups
        h_out, w_out = summed_spatial.shape[-2:]
        summed_spatial = summed_spatial.view(n, groups, in_per_group, h_out, w_out).sum(dim=2)  # (N, G, H_out, W_out)

        # Broadcast per-group result to all out_channels in that group
        out_channels = self.weight.shape[0]
        out_per_group = out_channels // groups
        summed_spatial = summed_spatial.unsqueeze(2).expand(n, groups, out_per_group, h_out, w_out)
        return summed_spatial.reshape(n, out_channels, h_out, w_out)

    def forward(self, x):
        # apply inline clamp+round first for fp prologue.
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        x = torch.clamp(x, min=self.act_b, max=self.act_b + q - s)
        x = torch.round((x - self.act_b) / s)

        # Integer-weight conv term (uses the stored integer codes unchanged)
        out_w = self._conv(x, self.weight)
        out_ones = self._conv1(x)

        output_weight_scale = self.weight_scale.reshape(1, -1, 1, 1)
        # Reduce weight_zero_point to one scalar per output channel
        zp = self.weight_zero_point
        zp_per_out = zp.view(-1)[: self.weight.shape[0]].view(1, -1, 1, 1)

        out = out_w * output_weight_scale + out_ones * zp_per_out
        out = out * self.input_scale

        if not torch.isclose(
            self.input_zero_point,
            torch.zeros_like(self.input_zero_point),
        ).all():
            # For dequantized weights we use the per-channel weight_scale without reshaping,
            # so it broadcasts correctly over [out_channels, in_channels, kH, kW].
            w_dq = self.weight * self.weight_scale + self.weight_zero_point
            out = out + self.input_zero_point * self._conv(
                torch.ones_like(x), w_dq
            )

        out = out + self.bias.view(1, -1, 1, 1)
        return out * self.post_scale.view(1, -1, 1, 1) + self.post_shift.view(1, -1, 1, 1)


class _ExactIntegerLinear(torch.nn.Module):
    """
    Exact integer-input replacement for NoisyLinear, with optional inlined
    activation quantization parameters so that activation and affine are a
    single module.
    """

    def __init__(
        self,
        module: NoisyLinear,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor | float,
        input_zero_point: torch.Tensor | float,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
    ):
        super().__init__()
        del module
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        self.register_buffer("bias", bias.detach().clone(), persistent=True)
        self.register_buffer(
            "input_scale",
            _to_scalar_buffer(input_scale, weight),
            persistent=True,
        )
        self.register_buffer(
            "input_zero_point",
            _to_scalar_buffer(input_zero_point, weight),
            persistent=True,
        )
        self.register_buffer("weight_scale", weight_scale.detach().clone(), persistent=True)
        self.register_buffer(
            "weight_zero_point",
            weight_zero_point.detach().clone().to(device=weight.device, dtype=weight.dtype),
            persistent=True,
        )

    def forward(self, x):
        # Linear path currently assumes the input is already integer-coded.
        dequantized_input = x * self.input_scale
        if not torch.isclose(
            self.input_zero_point,
            torch.zeros_like(self.input_zero_point),
        ).all():
            dequantized_input = dequantized_input + self.input_zero_point
        dequantized_weight = self.weight * self.weight_scale + self.weight_zero_point
        return F.linear(dequantized_input, dequantized_weight, self.bias)


def _get_noisy_conv(module):
    return module if isinstance(module, NoisyConv2d) else None


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
    """Return integer-valued weight codes together with their quantization parameters."""
    weight = module.weight.detach()
    scale = torch.exp2(module.log_wght_s.detach())
    module.Q.scale = scale

    if module.qscheme.name == "PER_CHANNEL":
        reduce_dims = tuple(range(1, weight.dim()))
        zero_point = weight.amin(dim=reduce_dims, keepdim=True)
    else:
        zero_point = weight.amin()

    module.Q.zero_point = zero_point
    return module.Q.quantize(weight), scale.detach().clone(), zero_point.detach().clone()


def _get_effective_bias(module: NoisyConv2d | NoisyLinear) -> torch.Tensor:
    """Return the effective bias used by the quantized layer."""
    out_features = module.out_channels if isinstance(module, NoisyConv2d) else module.out_features
    if module.bias is None:
        return torch.zeros(out_features, device=module.weight.device, dtype=module.weight.dtype)

    if isinstance(module, NoisyConv2d) and getattr(module, "quant_bias", False):
        s = torch.exp2(module.log_wght_s.detach())
        if module.qscheme.name == "PER_CHANNEL":
            zero_point = module.weight.detach().amin((1, 2, 3), keepdim=True)
        else:
            zero_point = module.weight.detach().amin()
        module.Q_b.scale = s.ravel()
        module.Q_b.zero_point = zero_point.ravel()
        module.Q_b.rnoise_ratio.data = torch.zeros_like(module._noise_ratio)
        return module.Q_b.dequantize(module.Q_b.quantize(module.bias.detach()))

    return module.bias.detach().clone()


def _make_exact_integer_affine_from_quantized(
    module: NoisyConv2d | NoisyLinear,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor | float,
    input_zero_point: torch.Tensor | float,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    post_scale: torch.Tensor | None = None,
    post_shift: torch.Tensor | None = None,
    activation_module: NoisyActLin | None = None,
) -> torch.nn.Module:
    if isinstance(module, NoisyConv2d):
        return _ExactIntegerConv2d(
            module,
            weight,
            bias,
            input_scale,
            input_zero_point,
            weight_scale,
            weight_zero_point,
            post_scale,
            post_shift,
            activation_module=activation_module,
        )
    return _ExactIntegerLinear(
        module, weight, bias, input_scale, input_zero_point, weight_scale, weight_zero_point
    )


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _first_tensor(value):
    for tensor in _iter_tensors(value):
        return tensor
    return None


def _get_model_root(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _get_batch_inputs(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _get_first_validation_batch(datamodule):
    if hasattr(datamodule, "setup"):
        for stage in ("test", "validate", "fit", None):
            try:
                datamodule.setup(stage=stage)
                break
            except TypeError:
                datamodule.setup(stage)
                break
            except Exception:
                continue

    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, dict):
        first_key = next(iter(val_loader))
        val_loader = val_loader[first_key]
    elif isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    return next(iter(val_loader))


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value

def restore_plain_validation_step(model: torch.nn.Module):
    """
    Remove GDNSQ validation decoration so transformed inference-only models
    can be validated without quantizer-width bookkeeping.
    """
    model.validation_step = type(model).validation_step.__get__(model, type(model))


def fuse_batchnorm_and_normalize_activation_scales(model: torch.nn.Module) -> tuple[int, int]:
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
    normalized_count = 0

    # Materialize the list of quantized affine sites first to avoid mutating
    # the module hierarchy (removing BatchNorm, replacing children) while
    # PyTorch is still traversing it via named_modules.
    affine_sites = list(_iter_quantized_affines(root))

    # Single logical pass over quantized affine layers:
    # for each one, optionally fuse a following BatchNorm and immediately
    # rewrite the node into an exact integer affine module with inlined activation.
    for name, parent, child_name, act, qmodule in affine_sites:
        quantized_weight, weight_scale, weight_zero_point = _get_quantized_weight_params(qmodule)
        effective_bias = _get_effective_bias(qmodule)
        ref = quantized_weight  # device/dtype reference
        current_scale = torch.exp2(act.log_act_s.detach()).to(device=ref.device, dtype=ref.dtype)
        current_zero_point = act.act_b.detach().to(device=ref.device, dtype=ref.dtype)
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
                        mean = batch_norm.running_mean.detach()
                        var = batch_norm.running_var.detach()
                        gamma = batch_norm.weight.detach()
                        beta = batch_norm.bias.detach()
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
                quantized_weight,
                effective_bias,
                current_scale,
                current_zero_point,
                weight_scale,
                weight_zero_point,
                post_scale,
                post_shift,
                activation_module=act,
            )
            parent._modules[child_name] = exact_affine

        normalized_count += 1
        logger.info(
            "Replaced fused activation in %s with inline round-clamp act and exact integer affine: act_scale=%s act_zero_point=%s.",
            name,
            current_scale.cpu().item() if current_scale.numel() == 1 else current_scale.flatten()[0].cpu().item(),
            current_zero_point.cpu().item() if current_zero_point.numel() == 1 else current_zero_point.flatten()[0].cpu().item(),
        )

    return fused_count, normalized_count


def print_weight_bias_stats(model: torch.nn.Module):
    """Print per-layer weight and bias value-sets for integer affine layers."""
    root = _get_model_root(model)
    for name, mod in root.named_modules():
        if isinstance(mod, (_ExactIntegerConv2d, _ExactIntegerLinear)):
            wvals = sorted(torch.unique(mod.weight.detach()).cpu().tolist())
            bvals = None
            if mod.bias is not None:
                bvals = sorted(torch.unique(mod.bias.detach()).cpu().tolist())
            logger.info(
                "layer=%s weight_values=%s bias_values=%s input_scale=%s input_zero_point=%s weight_scale=%s weight_zero_point_normalized=%s",
                name,
                wvals,
                bvals,
                float(mod.input_scale.detach().cpu().item()),
                float(mod.input_zero_point.detach().cpu().item()),
                sorted(torch.unique(mod.weight_scale.detach()).cpu().tolist()),
                sorted(torch.unique((mod.weight_zero_point / mod.weight_scale).detach()).cpu().tolist()),
            )


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)

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

    n_fused = fuse_batchnorm_and_normalize_activation_scales(qmodel)
    logger.info("Fused %d BatchNorm layer(s) into previous NoisyConv2d and removed from graph.", n_fused)
    restore_plain_validation_step(qmodel)

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
