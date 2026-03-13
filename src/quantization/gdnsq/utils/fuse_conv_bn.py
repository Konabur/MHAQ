import torch

from torch import nn
from operator import attrgetter
from src.aux.qutils import attrsetter
from src.quantization.gdnsq.gdnsq import BinaryQuantizer, Quantizer

def fuse_conv_bn(model: nn.Module, conv_name: str, bn_name: str):
    conv = attrgetter(conv_name)(model)

    W = conv.weight.clone()
    if conv.bias is not None:
        b = conv.bias.clone()
    else:
        b = torch.zeros(conv.out_channels, device=W.device)

    bn = attrgetter(bn_name)(model)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    scale = gamma / std
    shape = [-1] + [1] * (W.dim() - 1)

    conv.weight.data = W * scale.view(shape)
    conv.bias = nn.Parameter(beta + (b - mu) * scale)

    attrsetter(bn_name)(model, nn.Identity())  # Replacing bn module with Identity


# I could optimize the shit out of this function but since it is invocated only once, I don't care.
# So let's keep it KISS.
# TODO due to the questionable manipulations whom are mathematically correct (at first glance at least)
# There is slight metric loss. Because of the lack if fp32 precision I suppose.
# FIX IT!

def fuse_conv_bn_q(model: nn.Module, conv_name: str, bn_name: str):
    conv = attrgetter(conv_name)(model)
    prev_q = conv.Q
    new_zero_point = prev_q.zero_point + prev_q.scale.mul(0.5)
    new_scale = prev_q.scale

    W = conv.weight.detach().clone()
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(conv.out_channels, device=W.device)

    bn = attrgetter(bn_name)(model)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    bn_scale = gamma / std

    conv.scale = new_scale
    conv.zero_point = new_zero_point
    
    # hack with "shaking" weights in order to get proper binarization
    # (try to disable it and see what happens, or come up with a better solution)
    # conv.weight.data = conv.weight.data.double() + prev_q.scale.mul(0.5).to(conv.weight.device).double() - prev_q.scale.mul(
    # 0.5).to(conv.weight.device).double()
    # conv.weight.data = conv.weight.data.float()
    conv.weight.data = conv.weight.data + \
        prev_q.scale.mul(0.5).to(conv.weight.device) - \
        prev_q.scale.mul(0.5).to(conv.weight.device)
    conv.weight.data *= bn_scale.view([-1] + [1] * (W.dim() - 1))
    conv.bias = nn.Parameter(beta + (b - mu) * bn_scale)
    new_zero_point *= bn_scale.view(
        new_zero_point.shape).to(new_zero_point.device)
    new_scale *= bn_scale.view(new_scale.shape).to(new_zero_point.device)

    rnoise_ratio = float(torch.as_tensor(
        prev_q.rnoise_ratio).reshape(-1)[0])
    reference_q = Quantizer(
        module=conv,
        scale=prev_q.scale,
        zero_point=prev_q.zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )
    binary_q = BinaryQuantizer(
        module=conv,
        scale=new_scale,
        zero_point=new_zero_point,
        min_val=prev_q.min_val,
        max_val=prev_q.max_val,
        rnoise_ratio=rnoise_ratio,
        qnmethod=prev_q.qnmethod,
    )
    with torch.no_grad():
        reference_weight = reference_q.dequantize(
            reference_q.quantize(conv.weight.detach().to(new_scale.device))
        )
        binary_weight = binary_q.dequantize(
            binary_q.quantize(conv.weight.detach().to(new_scale.device))
        )
        mean_diff = (reference_weight - binary_weight).mean().item()
        mean_abs_diff = (reference_weight -
                            binary_weight).abs().mean().item()

    conv.Q = binary_q
    conv.binary_quantizer_weight_mean_diff = mean_diff
    conv.binary_quantizer_weight_mean_abs_diff = mean_abs_diff

    attrsetter(bn_name)(model, nn.Identity())
    return {
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
    }