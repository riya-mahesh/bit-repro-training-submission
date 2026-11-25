#!/usr/bin/env bash
# Creates one software configuration: a CUDA toolkit module, the PyTorch wheel built for it,
# and RepDL (the baseline) built against them. Run once per configuration; takes ~10 minutes.
#
#   export REPRO_WORK="/scratch/$USER/bit-repro-final/$(hostname -s)"
#   bash scripts/setup_env.sh cuda124        CUDA 12.4, PyTorch 2.6.0
#   bash scripts/setup_env.sh cuda126        CUDA 12.6, PyTorch 2.7.1
#   bash scripts/setup_env.sh cuda130        CUDA 13.0, PyTorch 2.9.0 (not for Volta GPUs)
#
# PYTHON selects the interpreter (default python3); newer PyTorch releases need Python >= 3.10.
# CUDA_MODULE overrides the module name; set it to "" where nvcc is already on PATH.
set -euo pipefail
source "$(dirname "$0")/common.sh"
TAG=${1:-}
case "$TAG" in
  cuda124) MOD=cuda-12.4; TORCH=2.6.0; WHEEL=cu124 ;;
  cuda126) MOD=cuda-12.6; TORCH=2.7.1; WHEEL=cu126 ;;
  cuda130) MOD=cuda-13.0; TORCH=2.9.0; WHEEL=cu130 ;;
  *) echo "usage: $0 cuda124|cuda126|cuda130"; exit 1 ;;
esac
CUDA_MODULE=${CUDA_MODULE-$MOD}
ENV_DIR="$REPRO_WORK/envs/$TAG"
load_module "$CUDA_MODULE"
command -v nvcc >/dev/null || { echo "nvcc not found: check the module name with 'module avail cuda'"; exit 1; }

if [ -e "$ENV_DIR" ]; then
  echo "Environment already exists: $ENV_DIR. Reuse it via REPRO_VENV; setup will not delete it."
  exit 1
fi
mkdir -p "$ENV_DIR"
"${PYTHON:-python3}" -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip wheel "setuptools<70"
pip install -q "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$WHEEL"
pip install -q numpy
python3 -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU visible:', torch.cuda.is_available())"
echo "nvcc: $(nvcc --version | sed -n 4p)"

# RepDL, pinned to the commit used in the report. Its build compiles CUDA code with this toolkit.
git clone -q https://github.com/microsoft/RepDL.git "$ENV_DIR/RepDL"
git -C "$ENV_DIR/RepDL" checkout -q acd95bb
# --no-build-isolation: RepDL's setup.py imports torch, so the build must see this environment.
if (cd "$ENV_DIR/RepDL" && pip install -q --no-build-isolation .); then
  (cd "$TMPDIR" && python3 -c "import repdl; print('RepDL installed')")
else
  echo "RepDL build failed; environment setup is incomplete."; exit 1
fi
python3 -c "
import sys, torch
if torch.__version__.split('+')[0] != '$TORCH' or (torch.version.cuda or '').replace('.', '') != '$WHEEL'[2:]:
    sys.exit('wrong torch in this environment: %s, CUDA %s' % (torch.__version__, torch.version.cuda))
"
{ echo "CUDA_MODULE='$CUDA_MODULE'"; echo "TORCH='$TORCH'"; echo "WHEEL='$WHEEL'"; } > "$ENV_DIR/config"
echo "Ready: $TAG in $ENV_DIR"
