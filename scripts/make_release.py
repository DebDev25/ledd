#!/usr/bin/env python
"""Build ONE archive containing everything the paper depends on.

What goes in, and why each piece is irreplaceable
-------------------------------------------------
  * the deployment bundle          -- the reported model, ready to run
  * metrics.csv for every run      -- the per-epoch curves. EVERY "mean over epochs"
                                      number in the paper is computed from these, and
                                      they cannot be recovered from a checkpoint.
  * train.log for every run        -- effective loss ratios, LR schedule, timings
  * every *.json                   -- thresholds, protocol metrics, faithfulness
  * splits.json (deduplicated)     -- 25 copies, 3 distinct; one per seed is enough
  * selected best.pth              -- optimiser state stripped by default (3x smaller)
  * the OOD image sets             -- hours of generation; regenerable but expensive
  * manifest.json                  -- sha256 and size of every file, plus the git SHA

What stays out: last.pth (resume-only), the processed GenImage archive (already a
Kaggle Dataset), and superseded pre-optimiser-fix baseline runs.

    python scripts/make_release.py --runs /kaggle/working/runs \
        --ood /kaggle/working/ood_sdxl /kaggle/working/ood_pixart_eq \
        --out /kaggle/working/ledd_release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

# best.pth worth archiving: the reported model, its seeds, the matched control,
# the two streams, and the three baselines actually reported in the paper.
DEFAULT_CKPT_RUNS = [
    "abl_concat",                 # <- the model the paper reports
    "concat_s1", "concat_s2",
    "ctrl_spatial_from_s1", "ctrl_spatial_from_s1_s1", "ctrl_spatial_from_s1_s2",
    "stage1_spatial", "stage2_frequency",
    "base_npr", "base_cnndetection2", "base_resnet502",
]


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_sha(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def strip_optimizer(src: str, dst: str) -> tuple[int, int]:
    """Keep weights and provenance, drop optimiser/scheduler/scaler/RNG state.

    Adam carries two moment tensors per parameter, so a checkpoint is roughly three
    times the model size. For archival we want to be able to LOAD the model, not to
    resume training from it, so the optimiser state is dead weight.
    """
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    lean = {k: ck[k] for k in ("model", "epoch", "global_step", "best_metric") if k in ck}
    lean["_stripped"] = "optimizer/scheduler/scaler/rng removed by make_release.py"
    torch.save(lean, dst)
    return os.path.getsize(src), os.path.getsize(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--bundle", default=None, help="default: <runs>/export/ledd_detector.pt")
    ap.add_argument("--ood", nargs="*", default=[], help="OOD image directories to include")
    ap.add_argument("--out", default="/kaggle/working/ledd_release")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--ckpt-runs", nargs="*", default=DEFAULT_CKPT_RUNS)
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--keep-optimizer", action="store_true",
                    help="archive full checkpoints (3x larger, only needed to resume training)")
    args = ap.parse_args()

    runs = args.runs
    bundle = args.bundle or os.path.join(runs, "export", "ledd_detector.pt")
    stage = tempfile.mkdtemp(prefix="ledd_release_")
    manifest: dict = {"git_sha": git_sha(args.repo), "source_runs": runs, "files": {}}
    sizes: dict[str, int] = {}

    def take(src: str, name: str, group: str) -> None:
        dst = os.path.join(stage, name)
        os.makedirs(os.path.dirname(dst) or stage, exist_ok=True)
        shutil.copy(src, dst)
        n = os.path.getsize(dst)
        manifest["files"][name] = {"bytes": n, "sha256": sha256(dst)[:32], "group": group}
        sizes[group] = sizes.get(group, 0) + n

    # ---- 1. the deployed model
    if os.path.exists(bundle):
        take(bundle, "model/ledd_detector.pt", "model")
    else:
        print(f"!! no bundle at {bundle} -- run scripts/export_model.py first", file=sys.stderr)

    # ---- 2. per-epoch curves, logs and metrics (small, irreplaceable)
    seen_splits: dict[str, str] = {}
    for run in sorted(os.listdir(runs)):
        d = os.path.join(runs, run)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            if f == "splits.json":
                # 25 copies, 3 distinct -- keep one of each
                h = sha256(p)[:16]
                if h in seen_splits:
                    continue
                seen_splits[h] = run
                take(p, f"splits/{run}.splits.json", "splits")
            elif f.endswith((".csv", ".log", ".json")):
                take(p, f"metrics/{run}__{f}", "metrics")
    for f in sorted(os.listdir(runs)):
        p = os.path.join(runs, f)
        if os.path.isfile(p) and f.endswith((".csv", ".json")):
            take(p, f"metrics/_root__{f}", "metrics")

    # ---- 3. checkpoints
    if not args.no_checkpoints:
        for run in args.ckpt_runs:
            src = os.path.join(runs, run, "best.pth")
            if not os.path.exists(src):
                print(f"   (skipped, missing: {run}/best.pth)")
                continue
            name = f"checkpoints/{run}.best.pth"
            if args.keep_optimizer:
                take(src, name, "checkpoints")
            else:
                tmp = os.path.join(stage, "_tmp.pth")
                a, b_ = strip_optimizer(src, tmp)
                os.makedirs(os.path.join(stage, "checkpoints"), exist_ok=True)
                shutil.move(tmp, os.path.join(stage, name))
                n = os.path.getsize(os.path.join(stage, name))
                manifest["files"][name] = {"bytes": n, "group": "checkpoints",
                                           "sha256": sha256(os.path.join(stage, name))[:32],
                                           "stripped_from_bytes": a}
                sizes["checkpoints"] = sizes.get("checkpoints", 0) + n
                print(f"   {run}: {a/1e6:.0f} MB -> {b_/1e6:.0f} MB")

    # ---- 4. OOD image sets
    for odir in args.ood:
        if not os.path.isdir(odir):
            print(f"   (skipped, missing: {odir})")
            continue
        tag = os.path.basename(odir.rstrip("/"))
        for root, _, files in os.walk(odir):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, odir)
                take(p, f"ood/{tag}/{rel}", "ood")

    # ---- 5. manifest + readme
    manifest["totals_mb"] = {k: round(v / 1e6, 1) for k, v in sizes.items()}
    json.dump(manifest, open(os.path.join(stage, "manifest.json"), "w"), indent=2)
    with open(os.path.join(stage, "README.txt"), "w") as f:
        f.write(
            "LEDD release archive\n"
            f"git {manifest['git_sha'][:12]}\n\n"
            "model/        the reported detector (concat fusion), threshold included.\n"
            "              Load with scripts/export_model.py's format; see app/README.md.\n"
            "metrics/      per-epoch curves (metrics.csv), training logs, calibration and\n"
            "              protocol JSON. Every mean-over-epochs figure in the paper comes\n"
            "              from these files and CANNOT be recomputed from a checkpoint.\n"
            "splits/       generator/image split assignment, deduplicated. val_generator and\n"
            "              test_generator are identical across all runs; only the 5% in-dist\n"
            "              holdout differs by seed, hence one file per seed.\n"
            "checkpoints/  best.pth, optimiser state stripped (load-only, not resume-able\n"
            "              unless --keep-optimizer was used).\n"
            "ood/          self-collected out-of-distribution image sets.\n"
            "manifest.json sha256 and size of every file above.\n")

    out = shutil.make_archive(args.out, "zip", stage)
    shutil.rmtree(stage, ignore_errors=True)
    print("\n" + "-" * 46)
    for k, v in sorted(manifest["totals_mb"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:9.1f} MB")
    print("-" * 46)
    print(f"  {'ARCHIVE':14s} {os.path.getsize(out)/1e6:9.1f} MB  {out}")
    print(f"  {len(manifest['files'])} files, git {manifest['git_sha'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
