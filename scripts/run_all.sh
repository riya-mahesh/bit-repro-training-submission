#!/usr/bin/env bash
# Full suite on cuda2, cuda3 and cuda5. On cuda4 use scripts/run_cuda4.sh.
# One configuration: CONFIGS=cuda126 bash scripts/run_all.sh
# Quick training with full correctness tests: CONFIGS=cuda126 QUICK=1 NCU=0 bash scripts/run_all.sh
set -euo pipefail
HOST=$(hostname -s)
case "$HOST" in
  cuda2|cuda3) DEFAULT_CONFIGS="cuda124 cuda126" ;;
  cuda5) DEFAULT_CONFIGS="cuda124 cuda126 cuda130" ;;
  *) echo "This launcher targets cuda2, cuda3 and cuda5; use the individual scripts elsewhere."; exit 1 ;;
esac
export REPRO_WORK=${REPRO_WORK:-/scratch/$USER/bit-repro-final/$HOST}
export RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
export GPU=${GPU:-0} MAX_JOBS=${MAX_JOBS:-4} NCU=${NCU:-1}
export PIP_NO_CACHE_DIR=1
source "$(dirname "$0")/common.sh"
cd "$REPO"
# RepDL and PyTorch builds need a compatible interpreter when creating an environment.
load_module "${PYTHON_MODULE-python-3.12}"
export PYTHON=${PYTHON:-$(command -v python3)}
nvidia-smi -i "$GPU"
echo "Use an idle GPU; this script does not terminate or manage other users' jobs."
for tag in ${CONFIGS:-$DEFAULT_CONFIGS}; do
  config_versions "$tag"
  if [ "$HOST" = cuda3 ] && [ "$tag" = cuda130 ]; then echo 'CUDA 13 cannot target TITAN V'; exit 1; fi
  (
    # Each configuration has an isolated shell so activation/module state cannot leak.
    unset REPRO_VENV CUDA_MODULE
    load_module "cuda-$EXPECT_CUDA"
    for candidate in \
        "${REPRO_ENV_ROOT:-$REPRO_WORK/envs}/$tag" \
        "/scratch/$USER/bit-repro-$HOST/envs/$tag" \
        "/scratch/$USER/bit-repro/envs/$tag"; do
      if [ -x "$candidate/bin/python3" ] && \
          "$candidate/bin/python3" -c "import torch, repdl; assert torch.__version__.split('+')[0] == '$EXPECT_TORCH'; assert torch.version.cuda == '$EXPECT_CUDA'" >/dev/null 2>&1; then
        export REPRO_VENV=$candidate; break
      fi
    done
    if [ -z "${REPRO_VENV:-}" ] && [ "$tag" = cuda124 ]; then
      candidate="/scratch/$USER/bit-repro/venv"
      if [ -x "$candidate/bin/python3" ] && "$candidate/bin/python3" -c "import torch, repdl; assert torch.__version__.split('+')[0] == '$EXPECT_TORCH'; assert torch.version.cuda == '$EXPECT_CUDA'" >/dev/null 2>&1; then
        export REPRO_VENV=$candidate
      fi
    fi
    if [ -z "${REPRO_VENV:-}" ]; then
      free_kb=$(df -Pk "$REPRO_WORK" | awk 'END {print $4}')
      if [ "$free_kb" -lt 10485760 ]; then
        echo "No complete $tag environment found; installation needs headroom (10 GiB required by this launcher)."
        echo "Set REPRO_ENV_ROOT to existing environments, or REPRO_WORK to a permitted filesystem with space."
        exit 1
      fi
      bash scripts/setup_env.sh "$tag"
      export REPRO_VENV="$REPRO_WORK/envs/$tag"
    fi
    echo "[$HOST $tag] Environment: $REPRO_VENV"
    bash scripts/run_tests.sh "$tag"
    bash scripts/run_experiments.sh "$tag"
  )
done
python3 scripts/summarize.py "$RESULTS_ROOT/$HOST" "$RESULTS_ROOT/$HOST/RESULTS_$RUN_ID.md"
echo "Completed. New results: $RESULTS_ROOT/$HOST ; run ID: $RUN_ID"
