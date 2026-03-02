from copy import deepcopy
from types import SimpleNamespace
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from torch import nn
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.loggers.default_logger import logger
from src.quantization.gdnsq.utils.fuse_conv_bn import fuse_conv_bn


class FuseConvBNCallback(Callback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @staticmethod
    def _fuse_model_conv_bn(model: nn.Module) -> int:
        layer_names, layer_types = zip(
            *[(name, type(module)) for name, module in model.named_modules()]
        )
        fused_pairs = 0

        for idx in range(len(layer_names) - 1):
            if not (
                issubclass(layer_types[idx], NoisyConv2d)
                and issubclass(layer_types[idx + 1], nn.BatchNorm2d)
            ):
                continue

            fuse_conv_bn(model, layer_names[idx], layer_names[idx + 1])
            fused_pairs += 1

        return fused_pairs
    
    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not hasattr(pl_module, "model"):
            logger.warning("Skipping FuseConvBNCallback: `pl_module.model` is missing.")
            return super().on_train_end(trainer, pl_module)

        quant_cfg = getattr(getattr(trainer, "config", None), "quantization", None)
        if quant_cfg is None:
            logger.warning("Skipping FuseConvBNCallback: quantization config is missing.")
            return super().on_train_end(trainer, pl_module)

        target_act_bit = quant_cfg.act_bit
        target_weight_bit = quant_cfg.weight_bit
        was_training = pl_module.training
        pl_module.eval()

        logger.info(
            "FuseConvBNCallback: fusing Conv-BN and calibrating with target bits "
            f"(a={target_act_bit}, w={target_weight_bit})."
        )

        fused_pairs = self._fuse_model_conv_bn(pl_module.model)
        logger.info(f"FuseConvBNCallback: fused {fused_pairs} Conv-BN pairs.")
        original_calibration = deepcopy(quant_cfg.calibration)

        try:
            if quant_cfg.calibration is None:
                quant_cfg.calibration = SimpleNamespace(
                    act_bit=target_act_bit, weight_bit=target_weight_bit
                )
            else:
                quant_cfg.calibration.act_bit = target_act_bit
                quant_cfg.calibration.weight_bit = target_weight_bit

            trainer.calibrate(
                model=pl_module,
                datamodule=getattr(trainer, "datamodule", None),
                verbose=False,
            )
        finally:
            quant_cfg.calibration = original_calibration

        if was_training:
            pl_module.train()

        return super().on_train_end(trainer, pl_module)
