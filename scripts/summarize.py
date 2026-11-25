#!/usr/bin/env python3
"""Writes RESULTS.md from every run record under results/ (standard library only).

    python3 scripts/summarize.py                 reads results/, writes RESULTS.md
    python3 scripts/summarize.py DIR OUT.md      other folder / output file
    python3 scripts/summarize.py --check RUN.json
                                                 one run: compares it with the reference hash
                                                 and all other runs of the same setting

Tables: reproducibility, accuracy, speed at batch size 100, breakdown by training phase,
batch-size scaling and peak GPU memory. Exits with status 1 if any of our runs differ from
another run with the same settings or from the reference hashes below.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--check" else None
ARGS = [] if CHECK else sys.argv[1:]
SRC = ARGS[0] if len(ARGS) > 0 else os.path.join(ROOT, "results")
DST = ARGS[1] if len(ARGS) > 1 else os.path.join(ROOT, "RESULTS.md")

# Final weights (after epoch 2) of our method, by (seed, batch size). These match the included
# archived runs. A mismatch requires investigation; this is not a universal hardware guarantee.
EXPECTED = {(0, 50): '2ecd0e5156ceba1eb8acf8fd58db1302516202c09438321ae2f3cc95347b552d', (0, 100): '6b8ce5843c58dcd10a95e3339e634b66f95989e2972cc475427c8fb7b83ab2f2', (0, 200): '8abbccf6ec9d21efde0dbb4803fa1ed196f734cec2a150bb3d06b94a1028143a', (0, 500): '6e2ad4e4cf7e3a4625d8514db8148f7479112890c6b2e6f790312909040f47f4', (0, 1000): '1ab462c2a0758fefa225c67dd4cf6a821e9ca2baf3c193f9a627c44b414cfb19', (1, 100): 'c955a8b6185235b4d2fa1de6b684f150f6a7609cdfd6d30a69546fa9b46dc13d', (2, 100): '20eea77696252215c07bc76b2b63b79b42595af41efc799e2a2727fc3cc243e5', (3, 100): '7ad32ff9c8b31c5e3aac33777daea93e3f75dde0c7f8841645e37c99bc1012a3', (4, 100): 'f5347082020d6df0f67e70afbe1c2e53891523fd08d2bce0922bb1dbf2bb9929'}

METHODS = [("repro", "Ours"), ("repdl", "RepDL"), ("stock", "Standard PyTorch")]
GPU_ORDER = ["TITAN V", "RTX 2080 Ti", "A100", "RTX 4070"]


def gpu(device):
    for g in GPU_ORDER:
        if g.replace("RTX ", "") in device:
            return g
    return device.replace("NVIDIA ", "")


def record(f):
    try:
        d = json.load(open(f))
    except (ValueError, OSError) as exc:
        raise ValueError("Unreadable run record: " + f) from exc
    if "mode" not in d or "hashes" not in d:
        raise ValueError("Invalid run record: " + f)
    if d.get("steps") != d.get("epochs", 0) * (60000 // d.get("batch", 1)):
        return None                      # a shortened test run, not a full training run
    if d.get("epochs") != 2 or d.get("slices") != 3 or d.get("lr") != 0.01:
        raise ValueError("Unsupported training protocol in " + f)
    if "weights_epoch_2" not in d["hashes"]:
        raise ValueError("Missing final-weight hash in " + f)
    d["gpu"] = gpu(d["device"])
    tag = d.get("tag") or ""
    if not re.fullmatch(r"cuda\d+", tag):  # a custom run label: use the CUDA version instead
        tag = "cuda" + (d.get("cuda") or "").replace(".", "")
    d["config"] = (d["gpu"], tag)
    d["file"] = os.path.abspath(f)
    return d


def load(source=SRC):
    files = sorted(glob.glob(os.path.join(source, "**", "mnist", "*.json"), recursive=True))
    return [d for d in map(record, files) if d]


def config_key(c):
    g, t = c
    return (GPU_ORDER.index(g) if g in GPU_ORDER else 99, g, t)


def cuda_name(tag):
    t = tag.replace("cuda", "")
    return t[:-1] + "." + t[-1] if t.isdigit() else tag


def fingerprint(d):
    # All recorded hashes, epoch test accuracies, and the final recorded loss.
    return json.dumps([d["hashes"], d.get("curve"), d.get("final_loss")], sort_keys=True)


def med(xs):
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    return "–" if x is None else ("%.*f" % (nd, x))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def check(path):
    # One line per run, printed by run_experiments.sh after each training run.
    d = record(path)
    if not d:
        sys.exit("not a complete training run: " + path)
    name = dict(METHODS)[d["mode"]]
    h = d["hashes"]["weights_epoch_2"]
    print("  %s, seed %d, batch %d: test accuracy %.2f%%, %.2f ms per step, final weights %s"
          % (name, d["seed"], d["batch"], 100 * d["test_accuracy"], d["step_ms_median"], h[:12]))
    sources = {os.path.abspath(SRC), os.path.abspath(os.environ.get("RESULTS_ROOT", os.path.join(ROOT, "runs")))}
    previous = {r["file"]: r for source in sources for r in load(source)}
    others = [r for r in previous.values() if r["file"] != d["file"] and r["mode"] == d["mode"]
              and r["seed"] == d["seed"] and r["batch"] == d["batch"]]
    same = [r for r in others if fingerprint(r) == fingerprint(d)]
    nconf = len({r["config"] for r in same})
    ok = True
    if d["mode"] == "repro" and (d["seed"], d["batch"]) in EXPECTED:
        ok = h == EXPECTED[(d["seed"], d["batch"])]
        print("    reference hash %s: %s" % (EXPECTED[(d["seed"], d["batch"])],
              "MATCH" if ok else "MISMATCH (the arithmetic differs from the report)"))
    if others and d["mode"] == "stock":
        print("    differs from %d of %d earlier runs of this setting"
              % (len(others) - len(same), len(others)))
    elif others:
        print("    identical to %d of %d earlier runs of this setting (%d GPU / CUDA configurations)"
              % (len(same), len(others), nconf))
        ok = ok and len(same) == len(others)
    if d["mode"] == "stock":
        if not others:
            print("    reproducibility: insufficient comparison data")
        elif len(same) == len(others):
            print("    recorded outputs match the available comparison runs")
        else:
            print("    recorded outputs differ across the available comparison runs")
    elif ok:
        print("    recorded outputs match available comparisons; reference check passed where applicable")
    else:
        print("    bit-reproducible: NO")
    sys.exit(0 if ok else 1)


def main():
    if CHECK:
        check(CHECK)
    runs = load()
    if not runs:
        sys.exit("no run records found under " + SRC)
    configs = sorted({d["config"] for d in runs}, key=config_key)
    timed = [d for d in runs if not d.get("profile")]
    md = ["# Results", "",
          "Generated by `scripts/summarize.py` from %d training runs in `%s`." % (len(runs), os.path.relpath(SRC, ROOT)),
          "Step times are medians over the steps of a run (the first five steps excluded), then the median",
          "over repeated runs. All runs train LeNet-5 on MNIST for 2 epochs with Adam (learning rate 0.01).", ""]

    md += ["## Configurations", ""]
    rows = []
    for c in configs:
        ds = [d for d in runs if d["config"] == c]
        rows.append([c[0], cuda_name(c[1]), ds[0]["torch"], ds[0].get("driver", ""), len(ds)])
    md += [table(["GPU", "CUDA", "PyTorch", "Driver", "Runs"], rows), ""]

    # ---------------- reproducibility ----------------
    md += ["## Reproducibility", "",
           "A setting is a (seed, batch size) pair. Runs agree only if every recorded hash (initial",
           "weights, weights after each epoch, logits), epoch test accuracies, and final loss match.", ""]
    ok = any(d["mode"] == "repro" for d in runs)
    rows = []
    for mode, name in METHODS:
        groups = collections.defaultdict(list)
        for d in runs:
            if d["mode"] == mode:
                groups[(d["seed"], d["batch"])].append(d)
        if not groups:
            continue
        n = sum(len(v) for v in groups.values())
        distinct = sum(len({fingerprint(d) for d in v}) for v in groups.values())
        same = distinct == len(groups)
        ref = ""
        if mode == "repro":
            wrong = [k for k, v in groups.items() if k in EXPECTED
                     and any(d["hashes"]["weights_epoch_2"] != EXPECTED[k] for d in v)]
            ref = "all match" if not wrong else "MISMATCH for %s" % sorted(wrong)
            ok = ok and same and not wrong
        if mode == "repdl":
            ok = ok and same
        verdict = "identical within every setting" if same else \
            ("%d different results in %d runs" % (distinct, n) if distinct < n else "every run different")
        rows.append([name, n, len(groups), len({d["config"] for v in groups.values() for d in v}), verdict, ref or "–"])
    md += [table(["Method", "Runs", "Settings", "Configurations", "Result", "Reference hashes"], rows), ""]

    # ---------------- accuracy ----------------
    md += ["## Accuracy", "", "Batch size 100, seeds 0–4, mean ± standard deviation in percent.", ""]
    rows = []
    for mode, name in METHODS:
        per = collections.defaultdict(list)
        for d in timed:
            if d["mode"] == mode and d["batch"] == 100:
                per[(d["config"], d["seed"])].append(d)
        if not per:
            continue
        first = {k: v[0] for k, v in per.items()}
        by_seed = collections.defaultdict(set)
        for (c, s), d in first.items():
            by_seed[s].add((d["train_accuracy"], d["test_accuracy"]))
        if all(len(v) == 1 for v in by_seed.values()):     # same on every configuration
            vals = [next(iter(by_seed[s])) for s in sorted(by_seed)]
        else:                                              # differs: one run per configuration and seed
            vals = [(d["train_accuracy"], d["test_accuracy"]) for d in first.values()]
        tr = [100 * a for a, _ in vals]
        te = [100 * b for _, b in vals]
        sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0
        rows.append([name, "%.2f ± %.2f" % (statistics.mean(tr), sd(tr)), "%.2f ± %.2f" % (statistics.mean(te), sd(te)), len(vals)])
    md += [table(["Method", "Train accuracy", "Test accuracy", "Values averaged"], rows), ""]

    # ---------------- speed ----------------
    def step(c, mode, batch, seed=0):
        return med([d["step_ms_median"] for d in timed
                    if d["config"] == c and d["mode"] == mode and d["batch"] == batch and d["seed"] == seed])

    md += ["## Speed at batch size 100", "", "Median time per training step in ms (seed 0).", ""]
    rows = []
    for c in configs:
        o, r, s = step(c, "repro", 100), step(c, "repdl", 100), step(c, "stock", 100)
        if o is None:
            continue
        rows.append([c[0], cuda_name(c[1]), fmt(o), fmt(r), fmt(s),
                     "–" if r is None else "%.1f×" % (r / o), "–" if s is None else "%.2f×" % (o / s)])
    md += [table(["GPU", "CUDA", "Ours", "RepDL", "Standard PyTorch", "Speedup over RepDL",
                  "Overhead over standard PyTorch"], rows), ""]

    # ---------------- phases ----------------
    prof = [d for d in runs if d.get("profile") and d.get("breakdown_ms")]
    if prof:
        md += ["## Breakdown by training phase", "", "Median ms per step, batch size 100 (runs with a synchronization between phases).", ""]
        rows = []
        for c in configs:
            for mode, name in METHODS:
                ds = [d for d in prof if d["config"] == c and d["mode"] == mode]
                if ds:
                    b = {k: med([d["breakdown_ms"].get(k, 0) for d in ds]) for k in ("forward", "backward", "optimizer")}
                    rows.append([c[0], cuda_name(c[1]), name, fmt(b["forward"]), fmt(b["backward"]), fmt(b["optimizer"])])
        md += [table(["GPU", "CUDA", "Method", "Forward", "Backward", "Optimizer"], rows), ""]

    # ---------------- batch scaling ----------------
    batches = sorted({d["batch"] for d in timed}) or sorted({d["batch"] for d in runs})
    if len(batches) > 1:
        md += ["## Batch-size scaling", "", "Median ms per step, ours / standard PyTorch (seed 0).", ""]
        cs = [c for c in configs if any(d["batch"] != 100 for d in timed if d["config"] == c)]
        rows = []
        for b in batches:
            rows.append([b] + ["%s / %s" % (fmt(step(c, "repro", b)), fmt(step(c, "stock", b))) for c in cs])
        big = batches[-1]
        rows.append(["Overhead at %d" % big] + [
            "–" if not (step(c, "repro", big) and step(c, "stock", big)) else "%.1f×" % (step(c, "repro", big) / step(c, "stock", big))
            for c in cs])
        md += [table(["Batch size"] + ["%s, CUDA %s" % (c[0], cuda_name(c[1])) for c in cs], rows), ""]

    # ---------------- memory ----------------
    md += ["## Peak GPU memory", "", "MiB, including the MNIST dataset held on the GPU (about 210 MiB).", ""]
    rows = []
    for c in configs:
        for b in (100, max(batches)):
            vals = [med([d["peak_memory_mib"] for d in timed if d["config"] == c and d["mode"] == m and d["batch"] == b])
                    for m, _ in METHODS]
            if any(v is not None for v in vals):
                rows.append([c[0], cuda_name(c[1]), b] + [fmt(v, 0) for v in vals])
        rows = [r for i, r in enumerate(rows) if r not in rows[:i]]
    md += [table(["GPU", "CUDA", "Batch size", "Ours", "RepDL", "Standard PyTorch"], rows), ""]

    md += ["## Protocol coverage", "",
           "Minimum counts for the full protocol, per included configuration. This does not verify profiler captures.", ""]
    coverage = []
    for c in configs:
        ds = [d for d in runs if d["config"] == c]
        missing = []
        for mode, _ in METHODS:
            for seed in range(5):
                n = sum(d["mode"] == mode and d["seed"] == seed and d["batch"] == 100
                        and not d.get("profile") for d in ds)
                if n < (3 if seed == 0 else 1): missing.append("%s seed%d" % (mode, seed))
            if not any(d["mode"] == mode and d["batch"] == 100 and d.get("profile") for d in ds):
                missing.append(mode + " phase timing")
        for mode in ("repro", "stock"):
            for b in (50, 200, 500, 1000):
                if sum(d["mode"] == mode and d["seed"] == 0 and d["batch"] == b
                       and not d.get("profile") for d in ds) < 2:
                    missing.append("%s batch%d" % (mode, b))
        coverage.append([c[0], cuda_name(c[1]), "complete" if not missing else "; ".join(missing)])
    md += [table(["GPU", "CUDA runtime", "Training protocol"], coverage), ""]
    md += ["## Scope and limitations", "",
           "CUDA labels refer to the PyTorch runtime. The included historical A100 CUDA 12.4 records used nvcc 12.6; see README.md.",
           "These tables establish agreement for the included settings, not every GPU, CUDA release or model.",
           "CUDA and PyTorch versions both change between configurations; their effects are not isolated.",
           "Accuracy after two epochs is not a general convergence guarantee. CPU training records are not included.",
           "Only completed training records contribute. Coverage above must be checked before calling a rerun complete.",
           "The included A100 batch-500 Nsight report covers weight gradients only; forward and filter-scale comparisons are missing.", ""]
    md += ["## Verdict", "", "Our included runs match the full reference hashes and recorded outputs within each setting, and included RepDL runs agree: **%s**." % ("yes" if ok else "NO"), ""]
    open(DST, "w").write("\n".join(md))
    print("wrote", DST, "| reproducible and matching reference hashes:", "yes" if ok else "NO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
