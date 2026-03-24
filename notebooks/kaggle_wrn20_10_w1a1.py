"""
Kaggle notebook script for WRN-20-10 w1a1 quantization pipeline.

Usage on Kaggle:
  1. Upload the MHAQ repo as a Kaggle dataset (or clone from GitHub).
  2. Set DATASET variable below to the model you want to train.
  3. Enable GPU accelerator (T4).
  4. Click "Save Version" → "Save and Run All (Commit)" to run overnight.
  5. Download checkpoints from the output after completion.

Each model takes ~3-6 hours on T4. One model per session (12h limit).
To run all 3 models, use 2 Kaggle sessions in parallel over 2 nights.
"""
import os
import subprocess
import sys
import shutil
import glob

# ──────────────────────────────────────────────────────────────────────
# CONFIG: pick ONE of "cifar10", "cifar100", "svhn"
# ──────────────────────────────────────────────────────────────────────
DATASET = "cifar10"
# ──────────────────────────────────────────────────────────────────────

CONFIGS = {
    "cifar10":  "config/gdnsq_config_wrn20_10_cifar10_w1a1.yaml",
    "cifar100": "config/gdnsq_config_wrn20_10_cifar100_w1a1.yaml",
    "svhn":     "config/gdnsq_config_wrn20_10_svhn_w1a1.yaml",
}

# Paths — copy repo from read-only input to writable working dir.
REPO_SRC = "/kaggle/input/mhaq"
REPO_DIR = "/kaggle/working/mhaq"
WORK_DIR = "/kaggle/working"
CONFIG   = CONFIGS[DATASET]

TRAIN_CKPT = os.path.join(WORK_DIR, f"wrn20_10_{DATASET}_trained.ckpt")
FUSED_CKPT = os.path.join(WORK_DIR, f"wrn20_10_{DATASET}_fused.ckpt")
BINAR_CKPT = os.path.join(WORK_DIR, f"wrn20_10_{DATASET}_binarized.ckpt")


# ── SETUP ────────────────────────────────────────────────────────────
os.environ["WANDB_MODE"] = "disabled"

if not os.path.exists(REPO_DIR):
    shutil.copytree(REPO_SRC, REPO_DIR)


def run(cmd, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=REPO_DIR, **kwargs)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)


# ── 0. Install deps ──────────────────────────────────────────────────
run("pip install -q -r requirements.txt")

# ── 1. Train (QAT) ──────────────────────────────────────────────────
# The trained checkpoint is saved by ModelCheckpoint callback into
# lightning_logs/ or the default checkpoint dir. We find the best one
# after training.
run(f"python scripts/gdnsq_q_config.py --config {CONFIG}")

# Find best checkpoint saved by ModelCheckpoint callback.
ckpt_pattern = os.path.join(REPO_DIR, "**", "gdnsq_checkpoint-*.ckpt")
ckpts = sorted(glob.glob(ckpt_pattern, recursive=True), key=os.path.getmtime)
if not ckpts:
    print("ERROR: no checkpoint found after training!")
    sys.exit(1)
best_ckpt = ckpts[-1]
print(f"Best checkpoint: {best_ckpt}")
shutil.copy2(best_ckpt, TRAIN_CKPT)

# ── 2. Fuse BatchNorm ───────────────────────────────────────────────
run(
    f"python scripts/gdnsq_q_fuse_batchnorm.py "
    f"--config {CONFIG} "
    f"--checkpoint {TRAIN_CKPT} "
    f"--output {FUSED_CKPT}"
)

# ── 3. Binarize edges ───────────────────────────────────────────────
run(
    f"python scripts/gdnsq_q_binarize_edges.py "
    f"--config {CONFIG} "
    f"--checkpoint {FUSED_CKPT} "
    f"--output {BINAR_CKPT}"
)

print(f"\nDone! Checkpoints in {WORK_DIR}:")
for f in [TRAIN_CKPT, FUSED_CKPT, BINAR_CKPT]:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
