import torch

from typing import Tuple
from torch import nn

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq_utils import QNMethod
from src.quantization.gdnsq.layers.gdnsq_act_lin import NoisyActLin

from src.aux.qutils import is_biased


class NoisyConv2d(NoisyActLin, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: str | int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
        rand_noise: bool = False,
        quant_bias: bool = False,
        signed: bool = True,
        disable: bool = False,
        act_init_s: float = -10,
        act_init_q: float = 10,
        qnmethod: QNMethod = QNMethod.AEWGS,
    ) -> None:
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_activation_quantization(
            init_s=act_init_s,
            init_q=act_init_q,
            signed=signed,
            disable=disable,
        )
        self._init_weight_quantization(
            qscheme=qscheme,
            log_s_init=log_s_init,
            rand_noise=rand_noise,
            qnmethod=qnmethod,
            per_channel_shape=(out_channels, 1, 1, 1),
            quant_bias=quant_bias,
            bias_log_shape=(1,),
        )

    def _weight_quantization_dims(self) -> tuple[int, ...]:
        return (1, 2, 3)

    def _apply_affine(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._conv_forward(input, weight, bias)

    def _get_affine_bias(self) -> torch.Tensor | None:
        return self.bias

    def extra_repr(self) -> str:
        bias = is_biased(self)
        # log_wght_s = self.log_wght_s.item()
        # noise_ratio = self._noise_ratio.item()

        log_wght_s = self.log_wght_s
        noise_ratio = self._weight_noise_ratio()

        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},\n"
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation},\n"
            f"groups={self.groups}, bias={bias}, log_wght_s_mean={log_wght_s.mean()},\n"
            f"noise_ratio={noise_ratio}, quantized_bias={self.quant_bias}"
        )
