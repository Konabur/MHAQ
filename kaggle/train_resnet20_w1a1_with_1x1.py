"""
Kaggle training script for ResNet20 CIFAR100 W1A1 with 1x1 conv quantization.

Usage on Kaggle:
1. Create new notebook with GPU P100
2. Run this script
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/Konabur/MHAQ.git"
BRANCH = "binarization"
REPO_DIR = "/kaggle/working/MHAQ"
CONFIG_PATH = "config/gdnsq_config_resnet20_cifar100_aewgs_w1a1.yaml"


def clone_repo():
    """Clone repository."""
    if os.path.exists(REPO_DIR):
        print(f"Repository already exists at {REPO_DIR}")
        return

    print(f"Cloning {REPO_URL} (branch: {BRANCH})...")
    subprocess.run(
        ["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR],
        check=True
    )
    print("Clone complete")


def install_dependencies():
    """Install dependencies with P100-compatible torch."""
    print("\nInstalling dependencies...")

    # Install torch 2.2.0 for P100 compatibility (compute capability 6.0)
    print("Installing PyTorch 2.2.0 (P100 compatible)...")
    subprocess.run([
        "pip", "install", "-q",
        "torch==2.2.0",
        "torchvision==0.17.0"
    ], check=True)

    # Install requirements.txt but skip torch/torchvision
    print("Installing other dependencies...")
    subprocess.run([
        "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt",
        "--no-deps"  # Don't install dependencies to avoid torch reinstall
    ], check=False)  # Don't fail if some packages conflict

    # Install critical missing packages
    subprocess.run([
        "pip", "install", "-q",
        "lightning==2.2.0",  # Compatible with torch 2.2
        "pytorchcv",
        "piq",
        "torchmetrics"
    ], check=True)

    print("Dependencies installed")


def train():
    """Run training."""
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    os.chdir(REPO_DIR)

    # Disable wandb
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"

    cmd = ["python", "scripts/gdnsq_q_train.py", "--config", CONFIG_PATH]
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Checkpoint saved in: {REPO_DIR}")
        print("=" * 80)
    else:
        print("\nTraining failed")
        sys.exit(1)


def main():
    print("ResNet20 CIFAR100 W1A1 Training (with 1x1 quantization)")
    print("=" * 80)

    try:
        clone_repo()
        install_dependencies()
        train()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
