#!/usr/bin/env bash
# Scale check on a rented 48 GB GPU (A40 / RTX A6000 / L40S).
#   bash run_runpod.sh quick   -> ~5 min smoke test, look at the output first
#   bash run_runpod.sh full    -> 1.5B, seeds 42 43 44, FFT + LoRA, runs in the background
# Progress:  tail -f results_runpod/log_1.5b_main.txt
# Finished runs are saved one by one in results_runpod/runs/; re-running skips them.
set -euo pipefail
cd "$(dirname "$0")"

pip install -q -r requirements.txt
pip uninstall -y -q torchao 2>/dev/null || true
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"     # keep model downloads on the persistent volume
mkdir -p results_runpod

case "${1:-}" in
  quick)
    python forgetting.py --model Qwen/Qwen2.5-1.5B --seeds 42 --quick --out results_runpod
    ;;
  full)
    nohup python forgetting.py --model Qwen/Qwen2.5-1.5B --seeds 42 43 44 --out results_runpod \
        >> results_runpod/log_1.5b_main.txt 2>&1 &
    echo "started in background (pid $!). Watch: tail -f results_runpod/log_1.5b_main.txt"
    ;;
  *)
    echo "usage: bash run_runpod.sh quick|full"; exit 1
    ;;
esac
