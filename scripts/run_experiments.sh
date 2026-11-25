#!/usr/bin/env bash
# Trains LeNet-5 on MNIST with the three methods and records everything the report uses.
#
#   bash scripts/run_experiments.sh cuda124            full protocol, ~20 minutes per GPU
#   QUICK=1 bash scripts/run_experiments.sh cuda124    one run per method, ~2 minutes
#   GPU=1   bash scripts/run_experiments.sh cuda124    use GPU 1 instead of GPU 0
#   NCU=1   bash scripts/run_experiments.sh cuda124    also profile the convolution weight-gradient
#                                                      kernels with Nsight Compute
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
# Output: results/<host>_<config>/mnist/*.json (one record per run), then run
#   python3 scripts/summarize.py      to write RESULTS.md from all results/ folders.
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_config "${1:?usage: $0 cuda124|cuda126|cuda130}"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
build_repro

OUT="$REPO/results/${HOST}_${TAG}"
mkdir -p "$OUT/mnist" "$OUT/logs"
{
  nvidia-smi -i "$GPU" --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  nvcc --version | sed -n 4p
  python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Flags' | sed 's/Flags:.*avx2.*/AVX2: yes/; s/Flags:.*/AVX2: no/' || true
} > "$OUT/environment.txt" 2>&1

MODES=(repro stock)
if (cd "$TMPDIR" && python3 -c 'import repdl' >/dev/null 2>&1); then MODES=(repro repdl stock)
else echo "RepDL is not installed in '$TAG'; skipping its runs"; fi

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
  command -v ncu >/dev/null || { echo "ncu (Nsight Compute) not found"; exit 1; }
  mkdir -p "$OUT/profiling"
  for b in 100 500; do
    echo "[$HOST $TAG] Nsight Compute, batch size $b"
    ncu --set full --kernel-name-base function --kernel-name 'regex:weight_grad_kernel|filter_scales_kernel' \
        --launch-skip 8 --launch-count 4 --export "$OUT/profiling/wgrad_batch$b" --force-overwrite \
        python3 "$REPO/repro_torch/mnist_repro.py" --mode repro --device cuda --batch "$b" --seed 0 --slices 3 \
        --max-steps 10 > "$OUT/logs/ncu_batch$b.log" 2>&1 || { echo "  FAILED: see $OUT/logs/ncu_batch$b.log"; FAILED=1; }
    ncu --import "$OUT/profiling/wgrad_batch$b.ncu-rep" --page details > "$OUT/profiling/wgrad_batch$b.txt" 2>&1 || true
  done
fi

if ((FAILED)); then echo "Some runs failed or were not reproducible: see the messages above and $OUT/logs"; exit 1; fi
echo "Done: $OUT. Next: python3 scripts/summarize.py"
