#!/bin/bash
# Crash-auto-restart wrapper for the SSH pilot arms.
# Host runs overcommit_memory=2 (strict commit accounting) with co-tenant memory
# spikes -> training can die on ENOMEM at any moment. This wrapper reruns with
# --resume until the script exits 0 AND the test-window prediction npz exists.
# Usage: run_resilient.sh L1|L2 [gpu-id]
set -u
SCRATCH="${SSH_WORK_DIR:-/scratch/zceehw4/ssh_pilot}"
PY="${SSH_PYTHON:-/scratch/zceehw4/envs/sevir_venv/bin/python}"
cd "$(dirname "$0")" || exit 1

TAG="${1:-L1}"
GPU_ID="${2:-2}"
case "$TAG" in
  L1) SCRIPT=train_L1.py ;;
  L2) SCRIPT=train_L2.py ;;
  *) echo "usage: $0 L1|L2"; exit 2 ;;
esac

for attempt in $(seq 1 8); do
  echo "[wrapper $TAG] attempt $attempt start $(date +%F_%T)" >> "$SCRATCH/wrapper_${TAG}.log"
  CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=4 "$PY" "$SCRIPT" \
      --gpu 0 --epochs 120 --tag "$TAG" --resume \
      >> "$SCRATCH/train_${TAG}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$SCRATCH/pred_${TAG}.npz" ]; then
    echo "[wrapper $TAG] DONE attempt $attempt $(date +%F_%T) rc=0 pred exists" >> "$SCRATCH/wrapper_${TAG}.log"
    exit 0
  fi
  echo "[wrapper $TAG] attempt $attempt rc=$rc, sleeping 90s" >> "$SCRATCH/wrapper_${TAG}.log"
  sleep 90
done
echo "[wrapper $TAG] FAILED after 8 attempts $(date +%F_%T)" >> "$SCRATCH/wrapper_${TAG}.log"
exit 1
