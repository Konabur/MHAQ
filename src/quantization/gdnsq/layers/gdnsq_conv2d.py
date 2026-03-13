import torch

from typing import Tuple
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq import Quantizer
from src.quantization.gdnsq.gdnsq_utils import QNMethod

from src.aux.qutils import attrsetter, is_biased


class NoisyConv2d(nn.Conv2d):
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
        qnmethod: QNMethod = QNMethod.AEWGS,
    ) -> None:
        super().__init__(
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
        self.qscheme = qscheme

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(
                torch.Tensor([log_s_init]), requires_grad=True
            )
            zero_point = self.weight.detach().amin()
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty((out_channels, 1, 1, 1)).fill_(log_s_init),
                requires_grad=True,
            )
            self.log_b_s = nn.Parameter(
                torch.empty(1).fill_(log_s_init), requires_grad=True
            )
            zero_point = self.weight.detach().amin((1, 2, 3), keepdim=True)
        self._noise_ratio = torch.nn.Parameter(torch.Tensor([1]), requires_grad=False)

        self.register_buffer("zero_point", zero_point, persistent=False)
        self.register_buffer(
            "scale", torch.exp2(self.log_wght_s.detach()).clone(), persistent=False
        )

        self.Q = Quantizer(
            self, self.scale, self.zero_point, -inf, inf, qnmethod=qnmethod
        )
        self.rand_noise = rand_noise
        self.quant_bias = quant_bias
        if self.quant_bias:
            self.Q_b = Quantizer(
                self, torch.exp2(self.log_b_s), 0, -inf, inf, qnmethod=qnmethod
            )
        self.inited = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            scale = torch.exp2(self.log_wght_s)
            if self.qscheme == QScheme.PER_CHANNEL:
                zero_point = self.weight.amin((1, 2, 3), keepdim=True)
            elif self.qscheme == QScheme.PER_TENSOR:
                zero_point = self.weight.amin()

            with torch.no_grad():
                self.inited = True
                self.scale.copy_(scale.detach())
                self.zero_point.copy_(zero_point.detach())
        else:
            if not self.inited:
                with torch.no_grad():
                    self.scale.copy_(torch.exp2(self.log_wght_s).detach())
                    if self.qscheme == QScheme.PER_CHANNEL:
                        self.zero_point.copy_(
                            self.weight.detach().amin((1, 2, 3), keepdim=True)
                        )
                    elif self.qscheme == QScheme.PER_TENSOR:
                        self.zero_point.copy_(self.weight.detach().amin())
                    self.inited = True
            scale = self.scale
            zero_point = self.zero_point

        self.Q.scale = scale
        self.Q.zero_point = zero_point
        self.Q.rnoise_ratio.data = (
            self._noise_ratio
            if self.rand_noise
            else torch.zeros_like(self._noise_ratio)
        )



        # if self.quant_bias:
        #     if self.training:
        #         self.Q_b.scale = s.ravel()
        #         self.Q_b.zero_point = min.ravel()
        #         self.Q_b.rnoise_ratio.data = (
        #             self._noise_ratio
        #             if self.rand_noise
        #             else torch.zeros_like(self._noise_ratio)
        #         )
        #     bias = self.Q_b.dequantize(self.Q_b.quantize(self.bias))
        # else:
        #     bias = self.bias

        bias = self.bias
        weight = self.Q.dequantize(self.Q.quantize(self.weight))

        # mask = torch.ones_like(weight)
        # mask[:, :, weight.shape[-2] // 2, weight.shape[-1] // 2] = 0.0
        # weight = weight * mask

        return self._conv_forward(input, weight, bias)

    def extra_repr(self) -> str:
        bias = is_biased(self)
        # log_wght_s = self.log_wght_s.item()
        # noise_ratio = self._noise_ratio.item()

        log_wght_s = self.log_wght_s
        noise_ratio = (
            self._noise_ratio
            if self.rand_noise
            else torch.zeros_like(self._noise_ratio)
        )

        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},\n"
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation},\n"
            f"groups={self.groups}, bias={bias}, log_wght_s_mean={log_wght_s.mean()},\n"
            f"noise_ratio={noise_ratio}, quantized_bias={self.quant_bias}"
        )
