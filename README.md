# Bit-Reproducible Neural-Network Training on GPUs

LeNet-5 training on MNIST whose final weights are bit-identical on every GPU and CUDA version,
without fixing the order of floating-point additions. Sums are computed exactly in integer
arithmetic and rounded once to FP32, exp and log are correctly rounded, and no kernel uses
atomic operations. The baselines are RepDL (fixed-order reproducible training) and standard
PyTorch.

**Instructions:** see [`readme.txt`](readme.txt) for the folder layout and the commands that
rebuild everything from source on the NYU CIMS GPU servers.
**Results:** [`RESULTS.md`](RESULTS.md) summarizes every run in [`results/`](results/), which holds the raw records of all experiments in the report.

```bash
export REPRO_WORK=/scratch/$USER/repro PYTHON=python3.12
bash scripts/setup_env.sh cuda124          # once per server, ~10 minutes
bash scripts/run_tests.sh cuda124          # kernel correctness tests
bash scripts/run_experiments.sh cuda124    # all MNIST experiments, ~20 minutes
python3 scripts/summarize.py               # writes RESULTS.md
```
