import torch

from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct
from src.loggers.default_logger import logger
from src.aux.traverse import previous_leaf
import numpy as np


class ObserverHook:
    def __init__(self) -> None:
        pass

    def __call__(self, layer_name=None, *args):
        raise NotImplementedError("You need to implement __call__ method!")


class MinMaxObserver(ObserverHook):
    def __init__(self) -> None:
        super().__init__()
        self._min_values = torch.tensor([])
        self._max_values = torch.tensor([])

    def __call__(self, module, input, output):
        return self._hook(module, input, output)

    def _hook(self, module, input, output) -> None:
        min_value = torch.min(input[0]).reshape(1)
        max_value = torch.max(input[0]).reshape(1)

        if (
            hasattr(module, "min_values")
            and hasattr(module, "max_values")
            and isinstance(module.min_values, torch.Tensor)
            and isinstance(module.max_values, torch.Tensor)
            and module.min_values.numel() > 0
            and module.max_values.numel() > 0
        ):
            module.min_values = torch.cat((module.min_values.to(min_value.device), min_value))
            module.max_values = torch.cat((module.max_values.to(max_value.device), max_value))
        else:
            module.min_values = min_value
            module.max_values = max_value


def apply_mean_stats_activations(module, abits=8, max_bits = 24):    
    for name, m in module.named_modules():

        if isinstance(m, NoisyAct):
            if (
                not hasattr(m, "min_values")
                or not hasattr(m, "max_values")
                or m.min_values.numel() == 0
                or m.max_values.numel() == 0
            ):
                logger.warning(f"Skipping activation calibration for '{name}': empty min/max stats.")
                continue

            min = m.min_values.min()
            max = m.max_values.max()
            device = m.log_act_s.device
            dtype = m.log_act_s.dtype

            logger.info(f"Min: {min}, Max: {max}, prev_leaf: {previous_leaf(module, name)}")

            m.min_values = torch.empty(0, device=device, dtype=dtype)
            m.max_values = torch.empty(0, device=device, dtype=dtype)

            if not m.log_act_q.requires_grad and not m.log_act_s.requires_grad:
                abits = max_bits

            if max - min > 0:
                # not zero width
                log_s = torch.log2((max - min) / (2**abits - 1))
                log_q = log_s + abits

                m.act_b = torch.nn.Parameter(min.detach().to(device=device, dtype=dtype).reshape(1), requires_grad=m.act_b.requires_grad)
                m.log_act_q = torch.nn.Parameter(log_q.detach().to(device=device, dtype=dtype).reshape(1), requires_grad=m.log_act_q.requires_grad)
                m.log_act_s = torch.nn.Parameter(log_s.detach().to(device=device, dtype=dtype).reshape(1), requires_grad=m.log_act_s.requires_grad)
            else:
                # pruned 
                m.log_act_q = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype), requires_grad=False)
                m.log_act_s = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype), requires_grad=False)
                m.act_b = torch.nn.Parameter(min.detach().to(device=device, dtype=dtype).reshape(1), requires_grad=False)


def apply_quantile_weights_s(module, wbits=8, max_bits = 24, qscheme="per-channel"):

    for name, m in module.named_modules():
        # TODO: qscheme
        if isinstance(m, (NoisyLinear, NoisyConv2d)):
            max_ = m.weight.detach().amax((1, 2, 3))
            min_ = m.weight.detach().amin((1, 2, 3))

            if not m.log_wght_s.requires_grad:
                wbits = max_bits

            # XXX: handle max-min == 0
            # log_s = torch.max(torch.tensor(m.log_wght_s), torch.log2((max - min) / (2**wbits - 1)).reshape(m.log_wght_s.shape))
            log_s = torch.max(m.log_wght_s, torch.log2((max_ - min_) / (2**wbits - 1)).reshape(m.log_wght_s.shape))
            # log_s = torch.max(m.log_wght_s.clone().detach().requires_grad_(True), torch.log2((max - min) / (2**wbits - 1)).reshape(m.log_wght_s.shape))

            #m.log_wght_s = torch.nn.Parameter(torch.full(m.log_wght_s.shape, log_s), requires_grad=m.log_wght_s.requires_grad) # per-tensor
            m.log_wght_s = torch.nn.Parameter(log_s, requires_grad=m.log_wght_s.requires_grad)

            logger.debug(f"{name}_q = {m.log_wght_s} ")
