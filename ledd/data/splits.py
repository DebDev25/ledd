"""Generator-level and image-level splits.

Protocol (Architecture doc S7.1) -- three disjoint generator roles:
  * train generators      -> training data
  * validation generator  -> checkpoint selection + hyperparameter tuning
  * test generators       -> touched exactly once, at the end

Selecting checkpoints on in-distribution validation does not track the headline
claim; selecting on the test generators leaks them. Hence the middle role.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class SplitSpec:
    train_generators: List[str]
    val_generator: str
    test_generators: List[str]
    val_fraction: float = 0.05      # held-out images within train generators
    seed: int = 42
    meta: Dict = field(default_factory=dict)

    def validate(self) -> None:
        roles = list(self.train_generators) + [self.val_generator] + list(self.test_generators)
        dupes = {g for g in roles if roles.count(g) > 1}
        if dupes:
            raise ValueError(f"Generator(s) in more than one role: {sorted(dupes)} -- this leaks.")


def _list_class(root: str, gen: str, label: str) -> List[str]:
    d = os.path.join(root, gen, label)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def build_index(root: str, generators: Sequence[str]) -> List[Tuple[str, int, str]]:
    """Return [(path, label, generator)] with label 0=real, 1=fake."""
    items: List[Tuple[str, int, str]] = []
    for gen in generators:
        for label_name, y in (("real", 0), ("fake", 1)):
            for p in _list_class(root, gen, label_name):
                items.append((p, y, gen))
    return items


def make_splits(root: str, spec: SplitSpec) -> Dict[str, List[Tuple[str, int, str]]]:
    spec.validate()
    rng = random.Random(spec.seed)

    train_items = build_index(root, spec.train_generators)
    rng.shuffle(train_items)
    n_val = int(len(train_items) * spec.val_fraction)
    in_dist_val, train = train_items[:n_val], train_items[n_val:]

    return {
        "train": train,
        "val_indist": in_dist_val,                                  # sanity only
        "val_generator": build_index(root, [spec.val_generator]),   # selection signal
        "test_generator": build_index(root, spec.test_generators),  # untouched
    }


def save_splits(splits: Dict[str, List[Tuple[str, int, str]]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: [list(t) for t in v] for k, v in splits.items()}, f)


def load_splits(path: str) -> Dict[str, List[Tuple[str, int, str]]]:
    with open(path) as f:
        raw = json.load(f)
    return {k: [tuple(t) for t in v] for k, v in raw.items()}
