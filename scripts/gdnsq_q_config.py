import os
import sys
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer, Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ quantization.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        help="Path to the configuration file (YAML).",
        # default="config/gdnsq_config_yolo11.yaml"
        # default="config/gdnsq_config_resnet20_old.yaml"
        # default="config/gdnsq_config_rfdn.yaml"
        default="config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)
    trainer = Trainer(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()

    logger.info(f"Validate Model before quantization:\n{model}")
    validator.validate(model, datamodule=data)

    qmodel = quantizer.quantize(model, in_place=True)

    # qmodel.strict_loading = False
    qmodel.load_state_dict(torch.load("logs/MHAQ/c86fef_2026-03-01_21_49/checkpoints/gdnsq_checkpoint-459-0.5783.ckpt")['state_dict'], strict=False)
    # qmodel.load_state_dict(torch.load("logs/MHAQ/8c8abb_2026-03-07_22_36/checkpoints/gdnsq_checkpoint-639-0.5629.ckpt")['state_dict'], strict=False)

    logger.info("Validate model after layers replacement")
    validator.validate(qmodel, datamodule=data)
       
    # logger.info("Calibrating model initial weights and scales")
    # validator.calibrate(qmodel, datamodule=data)

    # # Finetune model
    # trainer.fit(qmodel, datamodule=data)

    idx = trainer.callbacks.index([cb for cb in trainer.callbacks if "ModelCheckpoint" in cb.__class__.__name__][0])
    validator.callbacks[idx] = trainer.callbacks[idx]

    # validate
    # validator.validate(qmodel, datamodule=data, ckpt_path="best")

    logger.info("Fusing BATCH NORM")
    # fuse
    quantizer.fuse_conv_bn(qmodel)
    print(qmodel.binary_quantizer_weight_mean_diffs)
    print(f"SUM of mean_abs_diff {sum([qmodel.binary_quantizer_weight_mean_diffs[l]["mean_abs_diff"] for l in qmodel.binary_quantizer_weight_mean_diffs])}")
    print(f"SUM of mean_diff {sum([qmodel.binary_quantizer_weight_mean_diffs[l]["mean_diff"] for l in qmodel.binary_quantizer_weight_mean_diffs])}")

    logger.info("Validate after fusing")
    validator.validate(qmodel, datamodule=data)

    exit(0)

    # logger.info("Replace weights with sgn(weight)")

    # validator.config.quantization.calibration.act_bit = 1
    # validator.config.quantization.calibration.weight_bit = 1
    

    # validator.calibrate(qmodel, datamodule=data, ckpt_path="best")
    # validator.calibrate(qmodel, datamodule=data)
    # validator.test(qmodel, datamodule=data, ckpt_path="best")
    # validator.test(qmodel, datamodule=data)

    # validator.predict(qmodel, datamodule=data, ckpt_path="best")
    # validator.predict(qmodel, datamodule=data)

if __name__ == "__main__":
    main()
