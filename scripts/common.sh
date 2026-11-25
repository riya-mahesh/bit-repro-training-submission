# Shared by the scripts in this folder. Source it; do not run it.
#
# REPRO_WORK holds the Python environments, builds and caches. On CIMS it must be outside the
# home folder, whose quota (4 GB) is too small: export REPRO_WORK=/scratch/$USER/repro
: "${REPRO_WORK:?set REPRO_WORK to a scratch folder, for example: export REPRO_WORK=/scratch/\$USER/repro}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PIP_CACHE_DIR="$REPRO_WORK/pip-cache"
export TMPDIR="$REPRO_WORK/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # GPU numbers as nvidia-smi shows them
HOST=$(hostname -s)

# load_module NAME: loads an environment module. Scripts do not always inherit the shell's
# 'module' function, so the usual init files are tried. If modules are unavailable, nvcc must
# already be on PATH (for example after running 'module load cuda-12.4' in the shell).
load_module() {
  [ -n "$1" ] || return 0
  if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      # shellcheck disable=SC1090
      [ -f "$f" ] && source "$f" && break
    done
  fi
  if command -v module >/dev/null 2>&1; then module load "$1"
  else echo "note: 'module' unavailable; using the nvcc already on PATH"; fi
}

# load_config TAG: loads the CUDA module and Python environment made by setup_env.sh.
load_config() {
  TAG=$1
  ENV_DIR="$REPRO_WORK/envs/$TAG"
  [ -f "$ENV_DIR/config" ] || { echo "No environment '$TAG'. Create it first, e.g.: bash scripts/setup_env.sh cuda124"; exit 1; }
  # shellcheck disable=SC1091
  source "$ENV_DIR/config"
  load_module "$CUDA_MODULE"
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  command -v nvcc >/dev/null || { echo "nvcc not found after loading '$CUDA_MODULE'"; exit 1; }
  export RESULT_TAG=$TAG
}

# build_repro: builds the PyTorch extension for the selected GPU and this CUDA version, outside
# the source tree, and points REPRO_BUILD at it (mnist_repro.py imports it from there).
build_repro() {
  local key
  key=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | tr ' ' '_')
  export REPRO_BUILD="$REPRO_WORK/build/${TAG}_${key}"
  echo "Building the extension for $key with $(nvcc --version | sed -n 4p)"
  rm -rf "$REPRO_BUILD"
  mkdir -p "$REPRO_BUILD/repro_torch"
  (cd "$REPO/repro_torch" && python3 setup.py build_ext --force \
      --build-lib "$REPRO_BUILD" --build-temp "$REPRO_BUILD/objects") > "$REPRO_BUILD/build.log" 2>&1 \
    || { tail -n 30 "$REPRO_BUILD/build.log"; echo "Build failed; full log: $REPRO_BUILD/build.log"; exit 1; }
  cp "$REPO"/repro_torch/repro_torch/*.py "$REPRO_BUILD/repro_torch/"
  PYTHONPATH="$REPRO_BUILD" python3 -c 'import repro_torch; assert repro_torch.has_cuda, "built without CUDA"'
}
