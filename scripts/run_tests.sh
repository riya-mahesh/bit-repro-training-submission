#!/usr/bin/env bash
# Builds and runs the CUDA correctness suite. Run from the repository root with
# REPRO_WORK set and the selected environment installed.
#
#   bash scripts/run_tests.sh cuda126                 GPU tests
#   bash scripts/run_tests.sh cuda126 --cpu-proof     also the CPU proof of exp and log over all
#                                                     2^32 inputs (several more minutes)
#   GPU=1 bash scripts/run_tests.sh cuda126           use GPU 1 instead of GPU 0
#
# Each GPU test compares the kernels with an exact CPU implementation of the same definition,
# bit for bit, across algorithms, split sizes and repeated runs. The exp/log test hashes the
# outputs for all 2^32 float32 inputs; the hashes must equal the ones below on every GPU.
# Output: runs/<host>/<config>/<run-id>/tests/tests.log
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_config "${1:?usage: $0 cuda124|cuda126|cuda130 [--cpu-proof]}"
CPU_PROOF=0; [ "${2:-}" != "--cpu-proof" ] || CPU_PROOF=1
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

OUT="$OUT/tests"
mkdir -p "$(dirname "$OUT")" "$REPRO_WORK/build/$HOST/$TAG"
mkdir "$OUT" || { echo "Test output already exists: $OUT"; exit 1; }
BIN=$(mktemp -d "$REPRO_WORK/build/$HOST/$TAG/tests.XXXXXX")
LOG="$OUT/tests.log"
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
log "GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader)"
log "nvcc: $(nvcc --version | sed -n 4p)"

cd "$REPO"
FLAGS=(-O3 -std=c++17 -arch=native --fmad=false
       -Irepro_gemm_cuda -Irepro_sum_cuda -Irepro_math_cuda -Irepro_conv_cuda -Irepro_nn_cuda)
build() {   # build NAME SOURCES...
  local name=$1; shift
  nvcc "${FLAGS[@]}" "$@" -o "$BIN/$name" >> "$LOG" 2>&1 || { log "BUILD FAILED: $name (see $LOG)"; exit 1; }
}
log "=== building ==="
build test_repro_gemm    repro_gemm_cuda/repro_gemm.cu repro_gemm_cuda/test_repro_gemm.cu
build test_repro_conv    repro_gemm_cuda/repro_gemm.cu repro_conv_cuda/repro_conv.cu repro_conv_cuda/test_repro_conv.cu
build test_repro_sum     repro_sum_cuda/repro_sum.cu repro_sum_cuda/test_repro_sum.cu
build test_repro_softmax repro_sum_cuda/repro_sum.cu repro_math_cuda/repro_softmax.cu repro_math_cuda/test_repro_softmax.cu
build test_repro_nn      repro_nn_cuda/repro_nn.cu repro_nn_cuda/test_repro_nn.cu
build exhaustive_math    repro_math_cuda/exhaustive_math.cu

FAILED=()
for t in test_repro_gemm test_repro_conv test_repro_sum test_repro_softmax test_repro_nn; do
  log "=== $t ==="
  if "$BIN/$t" >> "$LOG" 2>&1; then log "PASS $t"; else log "FAIL $t"; FAILED+=("$t"); fi
done

log "=== exp and log over all 2^32 float32 inputs ==="
"$BIN/exhaustive_math" "$OUT/exp_log_hashes.txt" repro_expf repro_logf >> "$LOG" 2>&1 || FAILED+=(exhaustive_math)
declare -A WANT=([repro_expf]=4ccc7cddaea06b26 [repro_logf]=117fa2a6bbdea567)
for f in repro_expf repro_logf; do
  got=$(awk -v f="$f" '$1 == "SUMMARY" && $2 == f {print $3}' "$OUT/exp_log_hashes.txt" 2>/dev/null || true)
  if [ "$got" = "${WANT[$f]}" ]; then log "PASS $f hash $got"
  else log "FAIL $f hash ${got:-missing}, expected ${WANT[$f]}"; FAILED+=("$f"); fi
done

if ((CPU_PROOF)); then
  log "=== CPU proof: exp and log correctly rounded on all 2^32 inputs ==="
  g++ -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off -Irepro_math_cuda \
      repro_math_cuda/check_repro_math.cpp -o "$BIN/check_repro_math" >> "$LOG" 2>&1
  if "$BIN/check_repro_math" >> "$LOG" 2>&1; then log "PASS cpu proof"; else log "FAIL cpu proof"; FAILED+=(cpu_proof); fi
fi

if ((${#FAILED[@]})); then log "OVERALL: FAIL (${FAILED[*]}); details in $LOG"; exit 1; fi
log "OVERALL: PASS"
