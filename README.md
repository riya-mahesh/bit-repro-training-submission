# Bit-Reproducible Neural-Network Training on GPUs

This code trains LeNet-5 on MNIST so that the final weights are bit-identical on every GPU
and CUDA version, without fixing the order of floating-point additions. Every sum is
computed exactly in integer arithmetic and rounded once to FP32, exp and log are correctly
rounded, and the remaining operations (max pooling, Adam, initialization) follow fixed
rules. No kernel uses atomic operations. The baselines are RepDL (fixed-order reproducible
training) and standard PyTorch.

The scripts rebuild everything from source on the NYU CIMS GPU servers, run the correctness
tests, repeat the experiments of the report, and write [`RESULTS.md`](RESULTS.md).

**Quick check** on one CIMS GPU server, about 25 minutes (details in [section 3](#3-quick-start-on-cims-about-25-minutes)):

```bash
export REPRO_WORK=/scratch/$USER/repro PYTHON=python3.12
bash scripts/setup_env.sh cuda124                # install CUDA 12.4, PyTorch and RepDL (once per server)
bash scripts/run_tests.sh cuda124                # every kernel equals its exact CPU version: "OVERALL: PASS"
QUICK=1 bash scripts/run_experiments.sh cuda124  # train once per method: our weights hash to 6b8ce5843c58...
python3 scripts/summarize.py                     # verdict in RESULTS.md: "reproducible ... yes"
```

## 1. Folder layout

| Folder / file | Contents |
|---|---|
| `repro_gemm_cuda/` | Matrix multiplication (fully connected layers): power-of-two scales, 21-bit integers split into int8 digits, exact INT64 sums |
| `repro_conv_cuda/` | Convolution forward, weight gradient and input gradient, read directly from the image (no im2col) |
| `repro_sum_cuda/` | Exact sums of FP32 values (bias gradients, means) and the fixed warp-shuffle reduction tree used by every kernel |
| `repro_math_cuda/` | Correctly rounded exp and log, softmax and cross-entropy |
| `repro_nn_cuda/` | Max pooling (first maximum wins) and Adam (fixed IEEE operations) |
| `repro_torch/` | PyTorch layers on top of these kernels, and the MNIST experiment (`mnist_repro.py`), which also runs RepDL and standard PyTorch |
| `scripts/` | Setup, tests, experiments and the results summary |
| `results/` | The raw record of every run in the report, one folder per server and CUDA version (`a100_cuda124/profiling` has the Nsight Compute profile) |
| `RESULTS.md` | Summary tables of all runs in `results/` (written by `scripts/summarize.py`) |

Each kernel folder has the GPU code (`.cu`, `.cuh`), an exact CPU version of the same
definition (`*_cpu.h` or `*_math.cuh`) and a GPU test (`test_*.cu`) comparing the two bit
for bit.

## 2. Requirements

- An NVIDIA GPU of compute capability 6.1 or newer. On CIMS: cuda2 (RTX 2080 Ti), cuda3
  (TITAN V) or cuda5 (RTX 4070).
- A CUDA toolkit module (cuda-12.4, cuda-12.6 or cuda-13.0), Python 3.10 or newer with
  venv and pip, git, and internet access for pip, RepDL and the MNIST files.
- Several GB of disk per configuration, outside the 4 GB CIMS home quota (`/scratch`).
- Do not compile with `--use_fast_math`: the results depend on exact IEEE arithmetic.

The scripts install everything else: PyTorch, numpy and RepDL (`setup_env.sh`), and the
MNIST files, which are downloaded on the first run.

## 3. Quick start on CIMS (about 25 minutes)

The quick start checks the main claim of the report on one GPU: training with our kernels
gives exactly the same weights as all our earlier runs on other GPUs and CUDA versions.

### Step 0. Log in and prepare

```bash
ssh -J <netid>@access.cims.nyu.edu <netid>@cuda2.cims.nyu.edu    # or cuda3 / cuda5
tmux                                     # optional: keeps runs alive if ssh drops
nvidia-smi                               # check that GPU 0 is free (else use GPU=1)
cd bit-repro-training-submission         # the unzipped submission
export REPRO_WORK=/scratch/$USER/repro   # environments and builds go to scratch
export PYTHON=python3.12                 # any Python >= 3.10
```

### Step 1. Install the software (once per server, about 10 minutes)

```bash
bash scripts/setup_env.sh cuda124
```

- **Why:** creates a Python environment with CUDA 12.4, PyTorch 2.6.0, numpy and RepDL (the
  baseline) at the exact versions used in the report.
- **Expect:** the last line is `Ready: cuda124`.

### Step 2. Test every kernel (about 5 minutes)

```bash
bash scripts/run_tests.sh cuda124
```

- **Why:** each GPU kernel (matrix multiplication, convolution, sums, softmax, pooling, Adam)
  is compared bit for bit with an exact CPU implementation of the same arithmetic, on many
  shapes and inputs. Exp and log are evaluated on all 2^32 float32 inputs and their outputs
  hashed.
- **Expect:** one `PASS` line per test, the hashes `repro_expf 4ccc7cddaea06b26` and
  `repro_logf 117fa2a6bbdea567`, and the last line `OVERALL: PASS`.
- **Meaning:** every kernel computes exactly its definition on this GPU. The exp/log hashes
  are the same on every GPU, so these functions give identical bits everywhere. A `FAIL`
  names the kernel that differs (full log: `results/<host>_cuda124/tests/tests.log`).

### Step 3. Train LeNet-5 on MNIST once with each method (about 2 minutes)

```bash
QUICK=1 bash scripts/run_experiments.sh cuda124
```

- **Why:** trains for 2 epochs with seed 0 and batch size 100, with ours, RepDL and standard
  PyTorch, from the same data, batch order and initial weights.
- **Expect:** after each run, the script prints the test accuracy, the time per step, the
  hash of the final weights, and how this hash compares with the reference and with every
  earlier run of the same setting in `results/`. For example, on the TITAN V:

  ```
  [cuda3 cuda124] repro_quick_20261201_101500
    Ours, seed 0, batch 100: test accuracy 98.26%, 4.34 ms per step, final weights 6b8ce5843c58
      reference hash 6b8ce5843c58: MATCH
      identical to 35 of 35 earlier runs of this setting (8 GPU / CUDA configurations)
      bit-reproducible: yes
  [cuda3 cuda124] repdl_quick_20261201_101530
    RepDL, seed 0, batch 100: test accuracy 97.76%, 14.86 ms per step, final weights f3b851eed7b7
      identical to 32 of 32 earlier runs of this setting (8 GPU / CUDA configurations)
      bit-reproducible: yes
  [cuda3 cuda124] stock_quick_20261201_101600
    Standard PyTorch, seed 0, batch 100: test accuracy 97.91%, 2.05 ms per step, final weights a823d48e018c
      differs from 32 of 32 earlier runs of this setting
      not reproducible, as expected for standard PyTorch
  ```

- **Meaning:**
  - **Ours** must show `MATCH` and `bit-reproducible: yes`. The earlier runs were made on four
    GPUs (TITAN V, RTX 2080 Ti, A100, RTX 4070) with CUDA 12.4 to 13.0, so a new run that is
    identical to all of them shows bit-reproducibility across hardware and software. The
    comparison covers every recorded hash (initial weights, weights after each epoch,
    outputs) and every logged loss, not only the final weights.
  - **RepDL** is also reproducible, because it fixes the order of every sum, but slower:
    about 5 times on cuda2 and cuda3, less on cuda5.
  - **Standard PyTorch** is the fastest but gives different weights on almost every run,
    which is the problem this project addresses.
  - Test accuracy is about 98% for all three, so reproducibility costs no accuracy.

  The full log of each run is in `results/<host>_cuda124/mnist/<method>_quick_<time>.txt`.

### Step 4. Summarize

```bash
python3 scripts/summarize.py
cat RESULTS.md
```

- **Why:** combines the new runs with the included results of the report and checks them.
- **Expect:** the last line `Our runs are bit-reproducible and match the reference hashes:
  yes`. In the Reproducibility table, ours and RepDL are "identical within every setting"
  and standard PyTorch is "every run different".
- **Meaning:** the new runs agree bit for bit with all earlier runs of the same setting. The
  other tables (accuracy, speed, time per phase, batch-size scaling, memory) are those of
  the report, with the new runs' timings included ([section 5](#5-what-the-summary-reports)).

### Other CUDA versions

Repeat steps 1-4 with `cuda126` (CUDA 12.6, PyTorch 2.7.1) or `cuda130` (CUDA 13.0, PyTorch
2.9.0, cuda2 and cuda5 only, since CUDA 13 dropped the TITAN V). The expected output is the
same, including our hash `6b8ce5843c58...`, which shows that the result does not depend on
the CUDA version either. The same holds on another server.

The quick runs are saved under new names and never overwrite the included results. To
return to the submitted state, delete the `*_quick_*` files in `results/` and run
`python3 scripts/summarize.py` again.

## 4. Full reproduction (as in the report)

On each server and for each CUDA version, create the environment once and run the full
protocol (about 20 minutes per configuration):

```bash
export REPRO_WORK=/scratch/$USER/repro
export PYTHON=python3.12

bash scripts/setup_env.sh cuda124
bash scripts/run_tests.sh cuda124
bash scripts/run_experiments.sh cuda124

bash scripts/setup_env.sh cuda126     # and the same two run commands with cuda126
bash scripts/setup_env.sh cuda130     # cuda2 and cuda5 only: CUDA 13.0 has no Volta support
```

Configurations used in the report: TITAN V (cuda3) with CUDA 12.4 and 12.6, RTX 2080 Ti
(cuda2) with CUDA 12.4 and 12.6, RTX 4070 (cuda5) with CUDA 12.4, 12.6 and 13.0, and an
A100 on the NYU HPC cluster ([section 6](#6-the-a100-on-the-nyu-hpc-cluster)).

The full protocol on one GPU trains each method with seed 0 three times and seeds 1-4 once
at batch size 100, once more with a timer around the forward pass, backward pass and
optimizer, and ours and standard PyTorch twice at batch sizes 50, 200, 500 and 1000. All
methods use the same data, batch order and initial weights.

Optional kernel profile of the convolution weight gradient (needs Nsight Compute with access
to the GPU performance counters, which CIMS may not grant):

```bash
NCU=1 bash scripts/run_experiments.sh cuda124
```

After all runs, on any of the servers, `python3 scripts/summarize.py` summarizes everything in
`results/`. To combine runs from several servers, copy their `results/<host>_<config>/`
folders into one `results/` folder first.

A full run on a server and CUDA version that is already in `results/` (for example cuda3
with cuda124) replaces the included files of that folder. Keep a copy of `results/` first
to compare against the originals.

## 5. What the summary reports

`RESULTS.md` has one table per measurement:

- **Reproducibility:** whether runs with the same seed and batch size agree in every hash
  (initial weights, weights after each epoch, logits) and every logged loss, and whether our
  final weights match the reference hashes in `scripts/summarize.py`.
- **Accuracy** after 2 epochs (seeds 0-4).
- **Speed:** median time per training step at batch size 100, speedup over RepDL and
  overhead over standard PyTorch.
- **Breakdown** into forward, backward and optimizer time.
- **Batch-size scaling:** time per step at batch sizes 50 to 1000.
- **Peak GPU memory.**

Standard PyTorch is expected to give different weights in every run: it is the speed and
accuracy reference, not a reproducible method. RepDL is reproducible but slower.

## 6. The A100 on the NYU HPC cluster

There the GPU is used inside a CUDA development container, with an existing Python
environment that has PyTorch 2.6.0 (CUDA 12.4 wheel), numpy and RepDL. On a node with an
allocated A100, from the repository folder:

```bash
singularity exec --nv --bind "$PWD:/work" --bind /path/to/venv:/venv --pwd /work \
    /share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash --noprofile --norc

# inside the container
export REPRO_WORK=/work/.work REPRO_VENV=/venv
export CXX=/usr/bin/g++ CC=/usr/bin/gcc LDSHARED='/usr/bin/g++ -shared -pthread -B/usr/bin/'
export LD_PRELOAD=$(/usr/bin/g++ -print-file-name=libstdc++.so.6)
bash scripts/run_tests.sh cuda124
bash scripts/run_experiments.sh cuda124
python3 scripts/summarize.py
```

The compiler settings make the extension build with the container's compiler and load with
its C++ runtime. The container's nvcc (12.6) is newer than the wheel's CUDA (12.4), which
only produces a version warning. Results go to `results/<node>_cuda124/`.

## 7. Running pieces by hand

```bash
source $REPRO_WORK/envs/cuda124/bin/activate && module load cuda-12.4
cd repro_torch && python3 setup.py build_ext --inplace
python3 mnist_repro.py --mode repro          # ours
python3 mnist_repro.py --mode stock          # standard PyTorch
python3 mnist_repro.py --mode repdl          # RepDL
python3 mnist_repro.py --help                # seed, batch size, epochs, profiling
```

Without a GPU, `mnist_repro.py` also runs on the CPU (`--device cpu`) with the exact CPU
versions of the kernels, and gives the same hashes.
