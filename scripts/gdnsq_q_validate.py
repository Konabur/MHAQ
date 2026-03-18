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

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision('high')

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ validation.")
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
        help="Path to the checkpoint (.ckpt) to use for validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    _disable_tracing(config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()
    qmodel = quantizer.quantize(model, in_place=True)

    validator.validate(qmodel, datamodule=data, ckpt_path=args.checkpoint)

if __name__ == "__main__":
    main()
