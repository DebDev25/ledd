#!/usr/bin/env python
"""Rebuild a runs/ tree from a release archive made by make_release.py.

The archive is flattened for storage -- checkpoints under checkpoints/, per-run
metrics under metrics/<run>__<file>, and splits DEDUPLICATED because 25 runs share
only 3 distinct split assignments. This puts it all back where the scripts expect it.

Two things to know before relying on the result
-----------------------------------------------
1. Checkpoints were archived with optimiser state stripped (unless make_release.py
   was given --keep-optimizer). They load for evaluation, ensembling and export.
   They will NOT resume training -- the optimiser, scheduler, scaler and RNG state
   are gone. Attempting to resume will silently restart the optimiser instead.

2. splits.json is reconstructed by seed, using the naming convention: a run whose
   name ends in _s1 or _s2 used that seed's split, everything else used seed 42.
   This matches the measured grouping (19 runs on one hash, 3 on each of the other
   two). --verify re-hashes and will tell you if it ever stops matching.

    python scripts/restore_release.py --zip ledd_release.zip --out /kaggle/working
    python scripts/restore_release.py --zip ledd_release.zip --out /kaggle/working --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile


def seed_of(run: str) -> str:
    """Which split a run used, from its name. See the docstring caveat."""
    if run.endswith("_s1"):
        return "concat_s1"
    if run.endswith("_s2"):
        return "concat_s2"
    return "abl_concat"                      # seed 42: the large majority


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="/kaggle/working",
                    help="runs/ , ood/ and the bundle are written under here")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash restored files against manifest.json")
    ap.add_argument("--no-ood", action="store_true", help="skip the image sets")
    a = ap.parse_args()

    runs = os.path.join(a.out, "runs")
    os.makedirs(runs, exist_ok=True)
    stage = os.path.join(a.out, "_release_stage")
    shutil.rmtree(stage, ignore_errors=True)

    with zipfile.ZipFile(a.zip) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        if a.no_ood:
            members = [m for m in members if not m.startswith("ood/")]
        z.extractall(stage, members=members)

    manifest = {}
    mp = os.path.join(stage, "manifest.json")
    if os.path.exists(mp):
        manifest = json.load(open(mp))
        print(f"archive from git {manifest.get('git_sha','?')[:12]}, "
              f"{len(manifest.get('files', {}))} files")

    seen_runs, n = set(), 0

    # ---- checkpoints  ->  runs/<run>/best.pth
    cdir = os.path.join(stage, "checkpoints")
    for f in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        run = f.replace(".best.pth", "")
        d = os.path.join(runs, run)
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(cdir, f), os.path.join(d, "best.pth"))
        seen_runs.add(run); n += 1

    # ---- metrics  ->  runs/<run>/<file>   (and runs/<file> for _root__)
    mdir = os.path.join(stage, "metrics")
    for f in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
        run, _, name = f.partition("__")
        if not name:
            continue
        dst = runs if run == "_root" else os.path.join(runs, run)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(mdir, f), os.path.join(dst, name))
        if run != "_root":
            seen_runs.add(run)
        n += 1

    # ---- splits, re-expanded from the deduplicated copies
    sdir = os.path.join(stage, "splits")
    avail = {}
    for f in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        avail[f.replace(".splits.json", "")] = os.path.join(sdir, f)
    missing = set()
    for run in sorted(seen_runs):
        src = avail.get(seed_of(run))
        if src is None:
            missing.add(seed_of(run)); continue
        shutil.copy(src, os.path.join(runs, run, "splits.json")); n += 1
    if missing:
        print(f"!! split files absent from archive: {sorted(missing)} -- runs needing "
              f"them have no splits.json")

    # ---- deployment bundle and extras
    for rel, dst in [("model/ledd_detector.pt", os.path.join(runs, "export")),
                     ("extra/prompts.json", os.path.join(a.out, "ood_raw"))]:
        src = os.path.join(stage, rel)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, os.path.basename(rel))); n += 1

    # ---- ood image sets, back to their own top-level directories
    odir = os.path.join(stage, "ood")
    if os.path.isdir(odir) and not a.no_ood:
        for tag in sorted(os.listdir(odir)):
            dst = os.path.join(a.out, tag)
            if os.path.exists(dst):
                print(f"   {tag}: already present, left alone")
                continue
            shutil.move(os.path.join(odir, tag), dst)
            print(f"   {tag}: restored")

    # ---- verification
    if a.verify and manifest:
        bad = 0
        for name, meta in manifest["files"].items():
            p = os.path.join(stage, name)
            if not os.path.exists(p):
                continue
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:32]
            if h != meta.get("sha256"):
                print(f"   MISMATCH {name}"); bad += 1
        print(f"verify: {bad} mismatches" if bad else "verify: all hashes match")

    shutil.rmtree(stage, ignore_errors=True)
    present = sorted(d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d)))
    print(f"\nrestored {n} files into {runs}")
    print(f"{len(present)} run directories: {present}")
    print("\nNOTE: optimiser state was stripped at archive time -- these checkpoints "
          "evaluate and export, but cannot resume training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
