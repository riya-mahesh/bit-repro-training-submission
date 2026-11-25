# Shared setup; numerical code is not modified by these scripts.
: "${REPRO_WORK:?set REPRO_WORK to a writable workspace for environments and builds}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR:-1}
export PIP_CACHE_DIR="$REPRO_WORK/pip-cache"
export TMPDIR="$REPRO_WORK/tmp"
mkdir -p "$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
HOST=$(hostname -s)
export RESULTS_ROOT=${RESULTS_ROOT:-$REPO/runs}
export RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}

load_module() {
  [ -n "$1" ] || return 0
  if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      if [ -f "$f" ]; then source "$f"; break; fi
    done
  fi
  if command -v module >/dev/null 2>&1; then module load "$1"
  else echo "Using the toolkit already on PATH (no module command)."; fi
}

config_versions() {
  TAG=$1
  case "$TAG" in
    cuda124) EXPECT_TORCH=2.6.0; EXPECT_CUDA=12.4 ;;
    cuda126) EXPECT_TORCH=2.7.1; EXPECT_CUDA=12.6 ;;
    cuda130) EXPECT_TORCH=2.9.0; EXPECT_CUDA=13.0 ;;
    *) echo "Unknown configuration: $TAG"; return 1 ;;
  esac
}

load_config() {
  config_versions "$1"
  if [ -n "${REPRO_VENV:-}" ]; then
    ENV_DIR=$REPRO_VENV
    # For a container, set CUDA_MODULE='' and put the matching nvcc on PATH.
    load_module "${CUDA_MODULE-cuda-$EXPECT_CUDA}"
  else
    ENV_DIR="$REPRO_WORK/envs/$TAG"
    [ -f "$ENV_DIR/config" ] || { echo "Missing $ENV_DIR/config; run setup_env.sh $TAG"; return 1; }
    source "$ENV_DIR/config"
    load_module "$CUDA_MODULE"
  fi
  [ -x "$ENV_DIR/bin/python3" ] || { echo "Missing Python: $ENV_DIR"; return 1; }
  source "$ENV_DIR/bin/activate"
  export PATH="$ENV_DIR/bin:$PATH" PYTHONNOUSERSITE=1
  unset PYTHONHOME PYTHONPATH REPRO_CPU_ONLY
  hash -r
  export GPU=${GPU:-0} CUDA_VISIBLE_DEVICES=${GPU:-0}
  command -v nvcc >/dev/null || { echo "nvcc not found"; return 1; }
  python3 - "$EXPECT_TORCH" "$EXPECT_CUDA" <<'CHECK'
import re, subprocess, sys, torch
assert torch.__version__.split('+')[0] == sys.argv[1], torch.__version__
assert torch.version.cuda == sys.argv[2], torch.version.cuda
version = subprocess.check_output(['nvcc', '--version'], text=True)
assert re.search(r'release ' + re.escape(sys.argv[2]) + r'[,\s]', version), 'nvcc does not match the selected CUDA version'
assert torch.cuda.is_available(), 'No CUDA GPU visible'
print('Using', sys.executable, torch.__version__, torch.cuda.get_device_name(0))
CHECK
  export RESULT_TAG=$TAG
  OUT="$RESULTS_ROOT/$HOST/$TAG/$RUN_ID"
}

build_repro() {
  mkdir -p "$REPRO_WORK/build/$HOST/$TAG"
  export REPRO_BUILD
  REPRO_BUILD=$(mktemp -d "$REPRO_WORK/build/$HOST/$TAG/build.XXXXXX")
  mkdir -p "$REPRO_BUILD/repro_torch"
  echo "Building extension; log: $REPRO_BUILD/build.log"
  (cd "$REPO/repro_torch" && python3 setup.py build_ext --force \
      --build-lib "$REPRO_BUILD" --build-temp "$REPRO_BUILD/objects") > "$REPRO_BUILD/build.log" 2>&1 \
    || { tail -n 30 "$REPRO_BUILD/build.log"; return 1; }
  cp "$REPO"/repro_torch/repro_torch/*.py "$REPRO_BUILD/repro_torch/"
  PYTHONPATH="$REPRO_BUILD" python3 -c 'import repro_torch; assert repro_torch.has_cuda; print(repro_torch._backend.__file__)'
}
