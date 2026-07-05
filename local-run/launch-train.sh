#!/usr/bin/env bash
# Launch axolotl training on the current config with per-attempt logging.
# Re-run this script for each attempt; each run gets its own timestamped log
# under /root/logs/train/ and /root/logs/train/latest.log always points at
# the most recent attempt. Run it from an interactive shell (e.g. the tmux
# "train" window) so the pane stays open after a failure.
set -u

# /root/.venv-hf-wandb/bin and a dead /snap/bin/accelerate shadow the real venv
export PATH="/root/.venv/bin:$PATH"
# Marlin NVFP4 JIT must compile with torch's CUDA (13.0), not system CUDA 12.8
export CUDA_HOME="/root/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1

CONFIG=/root/configs/26b-a4b-moe-nvfp4-lora.yaml
LOG_DIR=/root/logs/train
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOG" "$LOG_DIR/latest.log"

{
  echo "=== axolotl train attempt started $(date -Is) ==="
  echo "config: $CONFIG"
  echo "log: $LOG"
} | tee "$LOG"

axolotl train "$CONFIG" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "=== axolotl train exited with status $status at $(date -Is) ===" | tee -a "$LOG"
exit "$status"
