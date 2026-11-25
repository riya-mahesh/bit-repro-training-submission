Bit-Reproducible Neural-Network Training on GPUs
================================================

This code trains LeNet-5 on MNIST so that the final weights are bit-identical on every GPU
and CUDA version, without fixing the order of floating-point additions. Every sum is
computed exactly in integer arithmetic and rounded once to FP32, exp and log are correctly
rounded, and the remaining operations (max pooling, Adam, initialization) follow fixed
rules. No kernel uses atomic operations.

The scripts below rebuild everything from source on the NYU CIMS GPU servers, run the
correctness tests, repeat the experiments of the report, and write RESULTS.md.


1. Folder layout
----------------
repro_gemm_cuda/   matrix multiplication (fully connected layers): power-of-two scales,
                   21-bit integers split into int8 digits, exact INT64 sums
repro_conv_cuda/   convolution forward, weight gradient and input gradient, read directly
                   from the image (no im2col)
repro_sum_cuda/    exact sums of FP32 values (bias gradients, means) and the fixed
                   warp-shuffle reduction tree used by every kernel
repro_math_cuda/   correctly rounded exp and log, softmax and cross-entropy
repro_nn_cuda/     max pooling (first maximum wins) and Adam (fixed IEEE operations)
repro_torch/       PyTorch layers on top of these kernels, and the MNIST experiment
                   (mnist_repro.py), which also runs RepDL and standard PyTorch
scripts/           setup, tests, experiments and the results summary
results/           records of every run (created by the scripts)
RESULTS.md         summary tables of all runs (written by scripts/summarize.py)

Each kernel folder has the GPU code (.cu, .cuh), an exact CPU version of the same
definition (*_cpu.h or *_math.cuh) and a GPU test (test_*.cu) comparing the two bit for bit.


2. Requirements
---------------
- An NVIDIA GPU of compute capability 6.1 or newer. On CIMS: cuda2 (RTX 2080 Ti),
  cuda3 (TITAN V) or cuda5 (RTX 4070).
- A CUDA toolkit module (cuda-12.4, cuda-12.6 or cuda-13.0), Python 3.10 or newer with
  venv and pip, git, and internet access for pip, RepDL and the MNIST files.
- Several GB of disk per configuration, outside the 4 GB CIMS home quota (/scratch).
- Do not compile with --use_fast_math: the results depend on exact IEEE arithmetic.


3. Quick start on CIMS (about 25 minutes)
-----------------------------------------
    ssh -J <netid>@access.cims.nyu.edu <netid>@cuda2.cims.nyu.edu    # or cuda3 / cuda5
    tmux                                     # optional: keeps runs alive if ssh drops

    # get the code (or unzip the submission) and enter it
    cd bit-repro-training-submission

    # all environments, builds and caches go to scratch (local to each server)
    export REPRO_WORK=/scratch/$USER/repro
    export PYTHON=python3.12                 # any Python >= 3.10

    # once per server: CUDA 12.4 + PyTorch 2.6.0 + RepDL (~10 minutes)
    bash scripts/setup_env.sh cuda124

    # correctness tests of every kernel (~5 minutes); must end with "OVERALL: PASS"
    bash scripts/run_tests.sh cuda124

    # one training run per method (~2 minutes), then the summary
    QUICK=1 bash scripts/run_experiments.sh cuda124
    python3 scripts/summarize.py
    cat RESULTS.md

Check that the GPU is free first (nvidia-smi). To use another GPU on the same server,
prefix the commands with GPU=1.

Expected outcome: RESULTS.md ends with "bit-reproducible and match the reference hashes:
yes". With seed 0, our final weights hash to 6b8ce5843c58... on every GPU and CUDA version
(see results/<host>_<config>/mnist/repro_seed0_run1.txt).


4. Full reproduction (as in the report)
---------------------------------------
On each server and for each CUDA version, create the environment once and run the full
protocol (~20 minutes per configuration):

    export REPRO_WORK=/scratch/$USER/repro
    export PYTHON=python3.12

    bash scripts/setup_env.sh cuda124
    bash scripts/run_tests.sh cuda124
    bash scripts/run_experiments.sh cuda124

    bash scripts/setup_env.sh cuda126     # and the same two run commands with cuda126
    bash scripts/setup_env.sh cuda130     # cuda2 and cuda5 only: CUDA 13.0 has no Volta support

Configurations used in the report: TITAN V (cuda3) with CUDA 12.4 and 12.6, RTX 2080 Ti
(cuda2) with CUDA 12.4 and 12.6, RTX 4070 (cuda5) with CUDA 12.4, 12.6 and 13.0, and an A100
on the NYU HPC cluster (section 6).

The full protocol on one GPU trains each method with seed 0 three times and seeds 1-4 once
at batch size 100, once more with a timer around the forward pass, backward pass and
optimizer, and ours and standard PyTorch twice at batch sizes 50, 200, 500 and 1000. All
methods use the same data, batch order and initial weights.

Optional kernel profile of the convolution weight gradient (needs Nsight Compute with
access to the GPU performance counters, which CIMS may not grant):

    NCU=1 bash scripts/run_experiments.sh cuda124

After all runs, on any of the servers (all results are in results/):

    python3 scripts/summarize.py

To combine runs from several servers, copy their results/<host>_<config>/ folders into
one results/ folder first.


5. What the summary reports
---------------------------
RESULTS.md has one table per measurement:
  - reproducibility: whether runs with the same seed and batch size agree in every hash
    (initial weights, weights after each epoch, logits) and every logged loss, and whether
    our final weights match the reference hashes in scripts/summarize.py;
  - accuracy after 2 epochs (seeds 0-4);
  - median time per training step at batch size 100, speedup over RepDL and overhead over
    standard PyTorch;
  - forward / backward / optimizer time;
  - time per step at batch sizes 50 to 1000;
  - peak GPU memory.

Standard PyTorch is expected to give different weights in every run: it is the speed and
accuracy reference, not a reproducible method. RepDL is reproducible but slower.


6. The A100 on the NYU HPC cluster
----------------------------------
There the GPU is used inside a CUDA development container, with an existing Python
environment that has PyTorch 2.6.0 (CUDA 12.4 wheel), numpy and RepDL. On a node with an
allocated A100, from the repository folder:

    singularity exec --nv --bind "$PWD:/work" --bind /path/to/venv:/venv --pwd /work \
        /share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash --noprofile --norc

    # inside the container
    export REPRO_WORK=/work/.work REPRO_VENV=/venv
    export CXX=/usr/bin/g++ CC=/usr/bin/gcc LDSHARED='/usr/bin/g++ -shared -pthread -B/usr/bin/'
    export LD_PRELOAD=$(/usr/bin/g++ -print-file-name=libstdc++.so.6)
    bash scripts/run_tests.sh cuda124
    bash scripts/run_experiments.sh cuda124
    python3 scripts/summarize.py

The compiler settings make the extension build with the container's compiler and load
with its C++ runtime. The container's nvcc (12.6) is newer than the wheel's CUDA (12.4),
which only produces a version warning. Results go to results/<node>_cuda124/.


7. Running pieces by hand
-------------------------
    source $REPRO_WORK/envs/cuda124/bin/activate && module load cuda-12.4
    cd repro_torch && python3 setup.py build_ext --inplace
    python3 mnist_repro.py --mode repro          # ours
    python3 mnist_repro.py --mode stock          # standard PyTorch
    python3 mnist_repro.py --mode repdl          # RepDL
    python3 mnist_repro.py --help                # seed, batch size, epochs, profiling

Without a GPU, mnist_repro.py also runs on the CPU (--device cpu) with the exact CPU
versions of the kernels, and gives the same hashes.
