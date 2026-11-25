#!/usr/bin/env bash
# Run from the repository root with REPRO_WORK set and the selected environment installed.
#
#   bash scripts/run_experiments.sh cuda126            40 training runs
#   QUICK=1 bash scripts/run_experiments.sh cuda126    one two-epoch run per method
#   GPU=1   bash scripts/run_experiments.sh cuda126    use GPU 1 instead of GPU 0
#   NCU=1   bash scripts/run_experiments.sh cuda126    profile forward, filter scales and weight gradients
#                                                     for both layers at batches 100 and 500
#
# Methods: repro (ours), repdl (RepDL, the fixed-order baseline), stock (standard PyTorch).
# All three use the same data, batch order and initial weights: LeNet-5, Adam with learning
# rate 0.01, 2 epochs (1,200 steps at batch size 100), FP32.
#
# Full protocol, on one GPU:
#   - seed 0 at batch size 100, three times per method        reproducibility, speed
#   - seeds 1-4 at batch size 100, once per method              reproducibility, accuracy
#   - one run per method timing forward, backward, optimizer    breakdown by phase
#   - batch sizes 50, 200, 500, 1000, twice each (ours, standard PyTorch)    batch scaling
# Output: runs/<host>/<config>/<run-id>/mnist/*.json; the script prints the command to summarize this run.
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_config "${1:?usage: $0 cuda124|cuda126|cuda130}"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
# All three methods are required for this experiment protocol.
python3 -c 'import repdl'
build_repro
mkdir -p "$OUT"
mkdir "$OUT/mnist" || { echo "Training output already exists: $OUT/mnist"; exit 1; }
mkdir -p "$OUT/logs"
printf 'RUNNING\n' > "$OUT/training_status.txt"
trap 'echo INCOMPLETE > "$OUT/training_status.txt"; exit 130' INT TERM
{
  nvidia-smi -i "$GPU" --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  nvcc --version | sed -n 4p
  python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Flags' | sed 's/Flags:.*avx2.*/AVX2: yes/; s/Flags:.*/AVX2: no/' || true
} > "$OUT/environment.txt" 2>&1

MODES=(repro repdl stock)

FAILED=0
train() {   # train LABEL MODE SEED BATCH [--profile]
  local label=$1 mode=$2 seed=$3 batch=$4; shift 4
  echo "[$HOST $TAG] $label"
  python3 "$REPO/repro_torch/mnist_repro.py" --mode "$mode" --device cuda --seed "$seed" --batch "$batch" \
      --epochs 2 --lr 0.01 --slices 3 --tag "$TAG" --out "$OUT/mnist/$label.txt" "$@" \
      > "$OUT/logs/$label.log" 2>&1 || { echo "  FAILED: see $OUT/logs/$label.log"; FAILED=1; return; }
  # accuracy, speed and final-weight hash, compared with the reference and all earlier runs
  python3 "$REPO/scripts/summarize.py" --check "$OUT/mnist/$label.json" || FAILED=1
}

if [ "${QUICK:-0}" = 1 ]; then
  for m in "${MODES[@]}"; do train "${m}_quick_$(date +%Y%m%d_%H%M%S)" "$m" 0 100; done   # never overwrites earlier runs
else
  for r in 1 2 3; do for m in "${MODES[@]}"; do train "${m}_seed0_run$r" "$m" 0 100; done; done
  for s in 1 2 3 4; do for m in "${MODES[@]}"; do train "${m}_seed$s" "$m" "$s" 100; done; done
  for m in "${MODES[@]}"; do train "${m}_phases" "$m" 0 100 --profile; done
  for b in 50 200 500 1000; do
    for r in 1 2; do for m in repro stock; do train "${m}_batch${b}_run$r" "$m" 0 "$b"; done; done
  done
fi

if [ "${NCU:-0}" = 1 ]; then
  PROFILE_REUSE_BUILD=1 bash "$REPO/scripts/profile_kernels.sh" "$TAG" || FAILED=1
fi

if ((FAILED)); then
  echo FAILED > "$OUT/training_status.txt"
  echo "Some runs failed or differed: see $OUT/logs"; exit 1
fi
echo COMPLETE > "$OUT/training_status.txt"
trap - INT TERM
echo "Done: $OUT"
printf 'Summarize this run with: python3 %q %q %q\n' "$REPO/scripts/summarize.py" "$OUT" "$OUT/RESULTS.md"
