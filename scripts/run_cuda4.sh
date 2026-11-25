#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != cuda4 ]]; then
  echo "Run this script on cuda4."
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export REPRO_WORK="${REPRO_WORK:-/scratch/$USER/bit-repro-cuda4}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
export GPU="${GPU:-0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PIP_NO_CACHE_DIR=1

# Nsight Compute does not support Maxwell.
export NCU=0
unset QUICK

source scripts/common.sh
cd "$REPO"

load_module "${PYTHON_MODULE:-python-3.12}"
export PYTHON="${PYTHON:-$(command -v python3.12)}"

nvidia-smi -i "$GPU"
df -h "$REPRO_WORK"

echo "Configurations: ${CONFIGS:-cuda124 cuda126}"
echo "Results root: $RESULTS_ROOT"
echo "Run ID: $RUN_ID"

for tag in ${CONFIGS:-cuda124 cuda126}; do
  case "$tag" in
    cuda124|cuda126) ;;
    *)
      echo "Unsupported configuration for this launcher: $tag"
      exit 1
      ;;
  esac

  (
    unset REPRO_VENV CUDA_MODULE
    config_versions "$tag"
    load_module "cuda-$EXPECT_CUDA"

    # Reuse an environment only if both packages import
    # and its PyTorch/CUDA versions match this configuration.
    for candidate in \
      "${REPRO_ENV_ROOT:-$REPRO_WORK/envs}/$tag" \
      "/scratch/$USER/bit-repro-cuda4/envs/$tag" \
      "/scratch/$USER/bit-repro/envs/$tag"
    do
      if [[ ! -x "$candidate/bin/python3" ]]; then
        continue
      fi

      if "$candidate/bin/python3" - \
          "$EXPECT_TORCH" "$EXPECT_CUDA" <<'PY'
import sys
import torch
import repdl

assert torch.__version__.split("+")[0] == sys.argv[1]
assert torch.version.cuda == sys.argv[2]
PY
      then
        export REPRO_VENV="$candidate"
        break
      fi
    done

    if [[ -z "${REPRO_VENV:-}" ]]; then
      if [[ -e "$REPRO_WORK/envs/$tag" ]]; then
        echo "Existing $tag environment is incomplete or incompatible:"
        echo "$REPRO_WORK/envs/$tag"
        echo "Stopping without changing or deleting it."
        exit 1
      fi

      free_kb=$(df -Pk "$REPRO_WORK" | awk 'END {print $4}')
      if (( free_kb < 10485760 )); then
        echo "Insufficient installation headroom for $tag."
        echo "This launcher requires 10 GiB free to create an environment."
        echo "Reuse one via REPRO_ENV_ROOT or choose another REPRO_WORK."
        exit 1
      fi

      bash scripts/setup_env.sh "$tag"
      export REPRO_VENV="$REPRO_WORK/envs/$tag"
    fi

    echo "[cuda4 $tag] Environment: $REPRO_VENV"

    # Stop immediately if correctness tests fail.
    bash scripts/run_tests.sh "$tag"

    # Full training, repeats, seeds, batch sizes and phase measurements.
    bash scripts/run_experiments.sh "$tag"
  )
done

"$PYTHON" scripts/summarize.py \
  "$RESULTS_ROOT/$HOST" \
  "$RESULTS_ROOT/$HOST/RESULTS_$RUN_ID.md"

echo "Completed: $RESULTS_ROOT/$HOST"
echo "Run ID: $RUN_ID"
