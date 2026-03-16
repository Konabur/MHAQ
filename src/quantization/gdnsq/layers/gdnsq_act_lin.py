import torch

from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq import Quantizer
from src.quantization.gdnsq.gdnsq_utils import QNMethod


class NoisyActLin(nn.Module):

    def _init_activation_quantization(
        self,
        init_s: float = -10,
        init_q: float = 10,
        noise_ratio: float = 1,
        disable: bool = False,
        qnmethod: QNMethod = QNMethod.STE,
    ) -> None:
        self.disable = disable

        zero_point = -torch.exp2(torch.tensor(init_q - 1).float())
        self._act_b = torch.tensor([zero_point]).float()
        self._log_act_s = torch.tensor([init_s]).float()
        self._log_act_q = torch.tensor([init_q]).float()
        self._noise_ratio = nn.Parameter(torch.tensor([noise_ratio]).float(), requires_grad=False)

        self.log_act_q = nn.Parameter(self._log_act_q, requires_grad=True)
        self.act_b = nn.Parameter(self._act_b, requires_grad=True)
        self.log_act_s = nn.Parameter(self._log_act_s, requires_grad=True)
        self.Q_input = Quantizer(
            self, torch.exp2(self._log_act_s), 0, -inf, inf, qnmethod=qnmethod
        )
        self.bw = torch.tensor(0.0)

    def _init_weight_quantization(
        self,
        *,
        qscheme: QScheme,
        log_s_init: float,
        rand_noise: bool,
        qnmethod: QNMethod,
        per_channel_shape: tuple[int, ...],
        quant_bias: bool = False,
        bias_log_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.qscheme = qscheme

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(torch.tensor([log_s_init]).float(), requires_grad=True)
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty(per_channel_shape).fill_(log_s_init), requires_grad=True
            )
        else:
            raise NotImplementedError(f"Unsupported qscheme {self.qscheme}")

        self.Q = Quantizer(
            self, torch.exp2(self.log_wght_s), 0, -inf, inf, qnmethod=qnmethod
        )
        self.rand_noise = rand_noise
        self.quant_bias = quant_bias

        if self.quant_bias:
            if bias_log_shape is None:
                raise ValueError("bias_log_shape must be provided when quant_bias=True")
            self.log_b_s = nn.Parameter(
                torch.empty(bias_log_shape).fill_(log_s_init), requires_grad=True
            )
            self.Q_b = Quantizer(
                self, torch.exp2(self.log_b_s), 0, -inf, inf, qnmethod=qnmethod
            )

    def _weight_quantization_dims(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _apply_affine(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_affine_bias(self) -> torch.Tensor | None:
        raise NotImplementedError

    def get_weight_minmax(self, keepdim: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qscheme == QScheme.PER_CHANNEL:
            dims = self._weight_quantization_dims()
            return (
                self.weight.amin(dims, keepdim=keepdim),
                self.weight.amax(dims, keepdim=keepdim),
            )
        if self.qscheme == QScheme.PER_TENSOR:
            return self.weight.amin(), self.weight.amax()
        raise NotImplementedError(f"Unsupported qscheme {self.qscheme}")

    def _configure_activation_quantizer(self) -> None:
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        zp = torch.round(self.act_b / s * 2) / 2 * s
        self.Q_input.zero_point = zp
        self.Q_input.min_val = zp
        self.Q_input.max_val = zp + q - s
        self.Q_input.scale = s

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable:
            return x

        self._configure_activation_quantizer()
        qx = self.Q_input.quantize(x)
        if not self.training:
            minmax = qx.aminmax()
            self.bw = torch.log2(minmax.max - minmax.min + 1)
        return qx

    def dequantize_input(self, qx: torch.Tensor) -> torch.Tensor:
        if self.disable:
            return qx
        return self.Q_input.dequantize(qx)

    def quantize_dequantize_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize_input(self.quantize_input(x))

    def _configure_weight_quantizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.exp2(self.log_wght_s)
        wmin, wmax = self.get_weight_minmax(keepdim=self.qscheme == QScheme.PER_CHANNEL)
        qwmin = torch.round(wmin / scale * 2) / 2 * scale
        qwmax = torch.round(wmax / scale * 2) / 2 * scale
        self.Q.scale = scale
        self.Q.zero_point = qwmin
        self.Q.min_val = qwmin
        self.Q.max_val = qwmax
        return scale, qwmin

    def quantize_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, zero_point = self._configure_weight_quantizer()
        return self.Q.quantize(self.weight), scale, zero_point

    def dequantize_weight(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        return qweight * scale + zero_point

    def quantize_dequantize_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qweight, scale, zero_point = self.quantize_weight()
        return self.dequantize_weight(qweight, scale, zero_point), scale, zero_point

    def _expand_weight_zero_point(self, zero_point: torch.Tensor) -> torch.Tensor:
        if zero_point.numel() == 1 or self.qscheme == QScheme.PER_CHANNEL:
            return zero_point.expand_as(self.weight)
        return zero_point

    def quantize_dequantize_bias(
        self,
        bias: torch.Tensor | None,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor | None:
        if bias is None or not getattr(self, "quant_bias", False):
            return bias

        self.Q_b.scale = scale.ravel()
        self.Q_b.zero_point = zero_point.ravel()
        self.Q_b.rnoise_ratio.data = self._weight_noise_ratio()
        return self.Q_b.dequantize(self.Q_b.quantize(bias))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self._get_affine_bias()
        input = self.quantize_dequantize_input(input)
        weight, scale, zero_point = self.quantize_dequantize_weight()
        bias = self.quantize_dequantize_bias(bias, scale, zero_point)
        return self._apply_affine(input, weight, bias)
