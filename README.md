# Bit-reproducible LeNet-5 training

This submission contains non-atomic CUDA kernels, PyTorch bindings, LeNet-5/MNIST
experiments, and 323 archived training records. The included runs agree in the
recorded hashes for our method across the eight GPU/software configurations below.
This is evidence for these tested configurations, not a guarantee for all hardware,
CUDA versions, inputs, or neural networks.

The convolution code is the original whole-batch non-atomic implementation used
for the archived measurements. It does **not** include the eight-image chunked
weight-gradient experiment from the earlier `Submit_version` directory. Do not mix
measurements from that implementation into this submission's performance tables.

## Train MNIST directly on CIMS

From the repository root in Bash, select the workspace containing your environment.
The example uses the launcher's default workspace and CUDA 12.6. If you installed
under another directory, use that directory as `REPRO_WORK`.

```bash
export REPRO_WORK="/scratch/$USER/bit-repro-final/$(hostname -s)"
module load python-3.12
export PYTHON="$(command -v python3.12)"
unset REPRO_VENV

if [ ! -d "$REPRO_WORK/envs/cuda126" ]; then
  bash scripts/setup_env.sh cuda126
fi

# Activate the environment and compile the extension for this session.
source scripts/common.sh
load_config cuda126 && build_repro
```

After a successful build, choose one command in the same terminal:

```bash
python3 repro_torch/mnist_repro.py --device cuda --mode repro
python3 repro_torch/mnist_repro.py --device cuda --mode stock
python3 repro_torch/mnist_repro.py --device cuda --mode repdl
python3 repro_torch/mnist_repro.py --help
```

For CUDA 12.4, use `cuda124` for both setup and loading. See
[CIMS_RUN_GUIDE.txt](CIMS_RUN_GUIDE.txt) for reusing a previous build, saving a
single run, and running the CUDA tests independently.

## Run on cuda2, cuda3, and cuda5

Enter the extracted repository directory on each CIMS server. You can use tmux to keep the run active after disconnecting.

```bash
nvidia-smi                           # choose a free GPU
bash scripts/run_all.sh
```

To choose the second GPU, use `GPU=1 bash scripts/run_all.sh`.
The launcher runs configurations sequentially:

| Server | GPU | Default configurations |
|---|---|---|
| cuda2 | RTX 2080 Ti | CUDA 12.4 / PyTorch 2.6.0; CUDA 12.6 / PyTorch 2.7.1 |
| cuda3 | TITAN V | CUDA 12.4 / PyTorch 2.6.0; CUDA 12.6 / PyTorch 2.7.1 |
| cuda5 | RTX 4070 | CUDA 12.4 / PyTorch 2.6.0; CUDA 12.6 / PyTorch 2.7.1; CUDA 13.0 / PyTorch 2.9.0 |

This reproduces the CIMS configuration matrix in the archived report. cuda2 can
optionally run CUDA 13.0 with a valid environment and compatible driver, but that
combination is not in the archived results. TITAN V cannot be compiled by CUDA 13.

On cuda4, use `bash scripts/run_cuda4.sh` for CUDA 12.4 and 12.6. It disables Nsight profiling.

The launcher:

1. Finds a complete existing environment under `REPRO_ENV_ROOT`, the new workspace,
   `/scratch/$USER/bit-repro-<host>/envs`, or `/scratch/$USER/bit-repro/envs`.
   It also recognizes cuda2's legacy CUDA 12.4 `bit-repro/venv`.
2. If no matching environment exists, creates one in the new workspace only when
   at least 10 GiB is free. This threshold provides installation headroom, not a
   guarantee against other users consuming shared disk space. It never deletes an
   existing environment. Low space or missing packages cause an actionable failure.
3. Verifies the selected Python, Torch runtime, nvcc version, and GPU visibility.
4. Runs GPU correctness tests, full training experiments, and Nsight profiling.
5. Checks each completed training record against the full archived reference hash
   and available runs, then writes a summary after all configurations finish.

Use an idle GPU and a valid allocation where required. The script does not stop
other jobs. Nsight performance-counter access must be enabled by the server.
Disk space is needed for builds even when environments already exist.

Optional overrides:

```bash
# Only one configuration; no reinstall if an existing environment is valid.
CONFIGS=cuda126 bash scripts/run_all.sh

# Existing environments in a different directory (contains cuda124/, cuda126/, ...).
REPRO_ENV_ROOT=/path/to/envs bash scripts/run_all.sh

# Use an approved filesystem with more space for new environments/builds.
REPRO_WORK=/path/with/space/bit-repro/cuda2 bash scripts/run_all.sh

# Explicitly skip Nsight if counters are unavailable. This leaves profiling incomplete.
NCU=0 bash scripts/run_all.sh
```

A full protocol has 40 training runs per configuration, plus correctness tests and
profiling. Runtime depends on hardware, contention, and profiler replay overhead;
there is no fixed completion-time promise. Detach from tmux with Ctrl+B, then D.

## Where results go

Packaged historical measurements remain unchanged under `results/`.
By default, suite executions write the following files under `runs/`. Set `RESULTS_ROOT` to select another directory.

```text
runs/<host>/<cuda-tag>/<UTC-run-id>/
  tests/tests.log
  mnist/*.json and *.txt
  logs/*.log
  environment.txt
  training_status.txt
  profiling/conv_batch100.ncu-rep and .txt
  profiling/conv_batch500.ncu-rep and .txt
  profiling/status.txt
```

Each launch has a new run ID. Existing run directories are not overwritten.
The launcher writes `runs/<host>/RESULTS_<run-id>.md`; it includes completed records
under that host, including earlier reruns, and lists minimum protocol coverage.
`COMPLETE` status plus the coverage table should be checked before treating a run
as complete. A crash may leave `RUNNING` or `INCOMPLETE`; neither means success.
Download the entire run directory. Environments/build caches are not result files.

To summarize only a selected results tree:

```bash
python3 scripts/summarize.py runs/cuda3 runs/cuda3/summary.md
```

To regenerate the archived report without a GPU or third-party Python packages:

```bash
python3 scripts/summarize.py
```

The summary ignores shortened training records. It reports incomplete protocol
coverage; matching hashes alone do not prove that every required run was collected.
Reference checks use full SHA-256 values; output may abbreviate hashes for display.
Hashes provide strong empirical evidence of agreement, not a mathematical proof of
collision-free equality or general correctness.

## Experiment definition

All modes use LeNet-5, MNIST, Adam with learning rate 0.01, two epochs, three slices
for our fixed-point layers, and seeds 0–4. Initialization and epoch permutations are
generated deterministically by the supplied code; this is **not** a no-shuffle run.
The same input order and initial weights are used across the three methods.

The full protocol runs seed 0 three times per method; seeds 1–4 once per method;
one synchronized phase-timing run per method; and batches 50, 200, 500 and 1000 twice
for our method and standard PyTorch. Baselines are RepDL (commit `acd95bb`) and
ordinary PyTorch. RepDL is required; the full protocol cannot silently skip it.

`RESULTS.md` reports runtime, phase timings, batch scaling, allocated GPU memory,
final accuracy, and agreement of hashes, epoch test accuracies, and final loss.
It does not compare every activation or every minibatch loss. Timing excludes the
first five steps and uses medians; phase-timing runs add synchronization and are
excluded from aggregate step timings. Peak memory includes the GPU-resident dataset
and is the PyTorch allocator's peak allocated memory, not total device usage.

## Evidence and limitations

The 323 JSON records and accompanying text were copied byte-for-byte from the
original non-atomic experiment archives.
The three convolution source files match the original repository exactly.

The archived matrix contains cuda2 (12.4, 12.6), cuda3 (12.4, 12.6), cuda5 (12.4,
12.6, 13.0), and A100 with a CUDA 12.4 PyTorch wheel. The latter's recorded compiler
was **nvcc 12.6.77**: its `cuda124` label denotes the runtime, not a matched 12.4
compiler/runtime configuration. Later A100 12.6/13.0 runs are not packaged here.

CUDA and PyTorch versions change together, so these experiments do not isolate the
effect of CUDA alone. Cross-device agreement is portability evidence; batch-size
experiments measure workload scaling, not distributed multi-GPU scaling. Accuracy
after two epochs is not a general convergence or accuracy-equivalence proof. CPU
training results and larger-model training are not included. Report performance
per GPU/configuration rather than one universal speedup. Shared-server contention
can affect timing, especially when comparing runs made at different times.

The packaged A100 text reports cover forward/GEMM and a general sample at batch
100, plus weight gradients at batches 100 and 500. They **do not** include batch-500
Conv1/Conv2 filter scales or Conv2 forward. CIMS Nsight binaries are not included.
These gaps must not be described as measured comparisons.

The updated profiling script captures both layers' forward, filter-scale, and
weight-gradient kernels at batches 100 and 500. It validates six launches and their
duration metrics; empty captures and report-export errors fail the run. These new
captures must actually be run on a GPU before claiming the gaps have been filled.
They are samples for diagnosis, not full training timelines or proof of the cause
of every timing trend. Do not use training times printed inside an ncu invocation
as normal benchmark timings.

## Profiling an existing CUDA 12.6 installation

Run from the repository root. Set `REPRO_WORK` to the workspace you used for
setup. For the default CIMS launcher workspace, use:

```bash
export REPRO_WORK="/scratch/$USER/bit-repro-final/$(hostname -s)"
unset REPRO_VENV
bash scripts/profile_kernels.sh cuda126
```

The script reads `$REPRO_WORK/envs/cuda126/config`, loads the CUDA module
recorded during setup, and activates `$REPRO_WORK/envs/cuda126`. If you
intentionally keep builds and environments in separate workspaces, set
`REPRO_VENV` to the existing venv directory before invoking the script.

For ordinary MNIST training, see [the standalone Python instructions](CIMS_RUN_GUIDE.txt)
(sections 1–3 and 5). The command above runs Nsight kernel profiling.

This builds the original non-atomic extension and captures the two batch sizes;
it does not repeat the 40-run training protocol. To run tests or training separately,
use `scripts/run_tests.sh cuda126` or `scripts/run_experiments.sh cuda126` with the
same variables. `NCU=1` adds profiling to the latter. Individual commands get separate
run IDs unless you explicitly export the same `RUN_ID` for them.

For A100/Singularity, allocate a GPU and use a development image/toolkit matching
the Torch wheel. Bind the source, workspace, and existing environment at their
original absolute paths (venv activation scripts can contain those paths), set
`CUDA_MODULE=''`, and use the individual scripts. The CIMS launcher is not an HPC
scheduler. Do **not** label a new build CUDA 12.4 while compiling with nvcc 12.6;
the scripts now reject that mismatch. Preparing HPC containers is site-specific.

## Code layout and local validation

`repro_gemm_cuda/`: fixed-point digit GEMM; `repro_conv_cuda/`: direct convolution;
`repro_sum_cuda/`: exact reductions; `repro_math_cuda/`: exp/log, softmax and loss;
`repro_nn_cuda/`: pooling and Adam; `repro_torch/`: bindings and MNIST training;
`scripts/`: environments, runners, checks and aggregation.

Kernel tests compare the GPU implementation with CPU implementations of the same
numerical definitions. Passing them does not mean quantized GEMM equals an exact
real-valued dot product. No numerical kernels or training arithmetic were changed
when preparing this package. Release validation on the local Mac covers scripts,
archived records and report generation; it is not a fresh CUDA execution.
