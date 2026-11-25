#!/usr/bin/env bash
# Targeted comparison of BOTH convolution layers at batches 100 and 500.
# Independent of the full training suite: bash scripts/profile_kernels.sh cuda126
# PROFILE_RESUME=1 with the original RUN_ID validates and reuses completed captures.
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_config "${1:?usage: $0 cuda124|cuda126|cuda130}"
command -v ncu >/dev/null || { echo "Nsight Compute (ncu) is missing"; exit 1; }
if [ "${PROFILE_REUSE_BUILD:-0}" != 1 ]; then build_repro; fi
: "${REPRO_BUILD:?compiled extension is required}"
mkdir -p "$OUT/logs"
if [ "${PROFILE_RESUME:-0}" = 1 ]; then
  mkdir -p "$OUT/profiling"
else
  mkdir "$OUT/profiling" || { echo "Profiling output exists: $OUT/profiling"; exit 1; }
fi
echo RUNNING > "$OUT/profiling/status.txt"
if [ "${PROFILE_RESUME:-0}" = 1 ]; then
  ncu --version > "$OUT/profiling/ncu_version_resume.txt"
else
  ncu --version > "$OUT/profiling/ncu_version.txt"
fi
for b in 100 500; do
  report="$OUT/profiling/conv_batch$b"
  if [ "${PROFILE_RESUME:-0}" = 1 ] && [ -s "$report.ncu-rep" ] && [ -s "$report.txt" ]; then
    if python3 "$REPO/scripts/check_profile.py" "$report.txt" "$OUT/logs/ncu_batch$b.log" "$b"; then
      echo "[$HOST $TAG] reusing validated batch-$b capture"
      continue
    fi
    echo FAILED > "$OUT/profiling/status.txt"
    echo "Existing capture is incomplete; preserved it. Choose a new RUN_ID to recapture."; exit 1
  fi
  if [ -e "$report.ncu-rep" ] || [ -e "$report.txt" ] || [ -e "$OUT/logs/ncu_batch$b.log" ]; then
    echo FAILED > "$OUT/profiling/status.txt"
    echo "Partial capture exists; preserved it. Choose a new RUN_ID to recapture."; exit 1
  fi
  echo "[$HOST $TAG] profiling both convolutions, batch $b"
  # Six matching launches per training step: two forwards, two scales, two weight gradients.
  # Skip five steps, then capture all six launches of the next step.
  if ! ncu --set full --kernel-name-base function \
      --kernel-name 'regex:forward_kernel|filter_scales_kernel|weight_grad_kernel' \
      --launch-skip 30 --launch-count 6 --export "$report" \
      python3 "$REPO/repro_torch/mnist_repro.py" --mode repro --device cuda \
      --batch "$b" --seed 0 --slices 3 --max-steps 10 > "$OUT/logs/ncu_batch$b.log" 2>&1; then
    echo FAILED > "$OUT/profiling/status.txt"
    echo "Capture failed: $OUT/logs/ncu_batch$b.log"; exit 1
  fi
  if ! ncu --import "$report.ncu-rep" --page details > "$report.txt" 2>&1; then
    echo FAILED > "$OUT/profiling/status.txt"
    echo "Report export failed: $report.txt"; exit 1
  fi
  if ! python3 "$REPO/scripts/check_profile.py" "$report.txt" "$OUT/logs/ncu_batch$b.log" "$b"; then
    echo FAILED > "$OUT/profiling/status.txt"; exit 1
  fi
done
echo COMPLETE > "$OUT/profiling/status.txt"
echo "Profiles: $OUT/profiling (diagnostic samples, not end-to-end benchmark timings)"
