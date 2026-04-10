"""
Kaggle training script for ResNet20 CIFAR100 W1A1 with 1x1 conv quantization.

Minimal version - assumes repo is cloned, skips requirements.txt installation.
Only installs critical missing packages if needed.

Usage on Kaggle:
1. Create new notebook with GPU P100
2. Clone repo: !git clone -b binarization https://github.com/Konabur/MHAQ.git
3. Run: !python MHAQ/kaggle/train_resnet20_w1a1_with_1x1.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
REPO_DIR = "/kaggle/working/MHAQ"
CONFIG_PATH = "config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml"

def check_dependencies():
    """Check and install only critical missing packages."""
    print("Checking dependencies...")

    missing = []
    try:
        import pytorch_lightning
    except ImportError:
        missing.append("pytorch-lightning")

    try:
        import pytorchcv
    except ImportError:
        missing.append("pytorchcv")

    try:
        import wandb
    except ImportError:
        missing.append("wandb")

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.run(
            ["pip", "install", "-q"] + missing,
            check=True
        )
    else:
        print("All critical packages available")


def verify_config():
    """Verify skip_1x1_conv is set to false."""
    config_file = Path(REPO_DIR) / CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    content = config_file.read_text()

    if "skip_1x1_conv: false" not in content and "skip_1x1_conv:false" not in content:
        print("Adding skip_1x1_conv: false to config...")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'qnmethod:' in line:
                indent = len(line) - len(line.lstrip())
                lines.insert(i + 1, ' ' * indent + 'skip_1x1_conv: false')
                break
        config_file.write_text('\n'.join(lines))
        print("Config updated")
    else:
        print("Config OK: skip_1x1_conv = false")


def train():
    """Run training."""
    print("\n" + "=" * 80)
    print("Starting training on P100...")
    print("Expected: ~1000 epochs, several hours")
    print("=" * 80 + "\n")

    os.chdir(REPO_DIR)

    cmd = ["python", "scripts/train.py", "--config", CONFIG_PATH]
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("Checkpoint saved in:", REPO_DIR)
        print("=" * 80)
    else:
        print("\nTraining failed or interrupted")
        sys.exit(1)


def main():
    print("ResNet20 CIFAR100 W1A1 Training (with 1x1 quantization)")
    print("=" * 80)

    if not Path(REPO_DIR).exists():
        print(f"\nError: Repository not found at {REPO_DIR}")
        print("Please clone first:")
        print("  !git clone -b binarization https://github.com/Konabur/MHAQ.git")
        sys.exit(1)

    try:
        check_dependencies()
        verify_config()
        train()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
