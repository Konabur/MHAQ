# Kaggle Training Script

## train_resnet20_w1a1_with_1x1.py

Minimal training script for ResNet20 CIFAR100 with 1-bit quantization including identity shortcuts (1x1 convolutions).

### Setup on Kaggle

1. Create new notebook
2. Settings → Accelerator → **GPU P100**
3. Clone repository:
   ```python
   !git clone -b binarization https://github.com/Konabur/MHAQ.git
   ```
4. Run training:
   ```python
   !python MHAQ/kaggle/train_resnet20_w1a1_with_1x1.py
   ```

### What it does

- Checks for critical missing packages (pytorch-lightning, pytorchcv, wandb)
- Verifies `skip_1x1_conv: false` in config
- Trains ResNet20 on CIFAR100 with 1-bit quantization
- Identity shortcuts (1x1 conv) are quantized

### Expected results

- Training time: ~8-15 hours on P100
- Final accuracy: ~53-58% top-1
- Checkpoint: `/kaggle/working/MHAQ/gdnsq_checkpoint-XXXX-X.XXXX.ckpt`

### After training

Download checkpoint and run locally:

```bash
# Fuse BatchNorm
python scripts/gdnsq_q_fuse.py \
  --config config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml \
  --checkpoint gdnsq_checkpoint-XXXX-X.XXXX.ckpt \
  --output fused_with_1x1.ckpt

# Export to int-only
python scripts/gdnsq_q_export_int_only.py \
  --config config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml \
  --checkpoint fused_with_1x1.ckpt \
  --output int_only_with_1x1.ckpt \
  --report int_only_with_1x1_report.json
```

Expected: ~3 float modules instead of 7 (init_block.conv, init_block.bn, output).
