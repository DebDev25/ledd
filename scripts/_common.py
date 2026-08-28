"""Shared CLI plumbing: puts the repo root on sys.path and parses config+overrides."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ledd.utils.config import apply_overrides, load_config  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="e.g. --set train.lr_new=1e-4 model.frequency.n_bands=16")
    return ap


def get_config(args) -> dict:
    return apply_overrides(load_config(args.config), args.set)
