"""Datasets and loaders.

The dataset returns the RGB tensor only; the FFT is computed inside the model
(on GPU, batched) rather than in the dataloader -- keeps CPU workers cheap on
free-tier machines and guarantees the spectrum is taken AFTER augmentation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .degradations import RandomDegradation, apply_named

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).float().div_(255.0)
    arr = arr.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (arr - mean) / std


class GenImageDataset(Dataset):
    """Items are (path, label, generator) triples from splits.py."""

    def __init__(
        self,
        items: Sequence[Tuple[str, int, str]],
        train: bool = False,
        augment: Optional[RandomDegradation] = None,
        hflip: float = 0.5,
        degradation: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.items = list(items)
        self.train = train
        self.augment = augment
        self.hflip = hflip
        self.degradation = degradation      # fixed named degradation for eval
        self.transform = transform or to_tensor_normalized
        self.generators = sorted({g for _, _, g in self.items})
        self._gen_to_idx = {g: i for i, g in enumerate(self.generators)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        path, label, gen = self.items[i]
        img = Image.open(path).convert("RGB")

        if self.train:
            import random

            if random.random() < self.hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.augment is not None:
                img = self.augment(img)
        elif self.degradation and self.degradation != "none":
            img = apply_named(img, self.degradation)

        return {
            "image": self.transform(img),
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": torch.tensor(self._gen_to_idx[gen], dtype=torch.long),
        }


class BalancedGeneratorBatchSampler(torch.utils.data.Sampler):
    """Every batch contains reals + fakes from SEVERAL generators.

    This matters more than the contrastive loss itself: if a batch's fakes all come
    from one generator, "fakes cluster together" degenerates into "this generator's
    fakes cluster together" -- the exact shortcut the cross-generator contrastive
    loss is meant to prevent.
    """

    def __init__(self, items: Sequence[Tuple[str, int, str]], batch_size: int,
                 min_generators: int = 3, seed: int = 42, drop_last: bool = True):
        self.batch_size = batch_size
        self.min_generators = min_generators
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.real_by_gen: Dict[str, List[int]] = {}
        self.fake_by_gen: Dict[str, List[int]] = {}
        for idx, (_, label, gen) in enumerate(items):
            (self.fake_by_gen if label == 1 else self.real_by_gen).setdefault(gen, []).append(idx)
        self.gens = sorted(set(self.real_by_gen) | set(self.fake_by_gen))
        self.n = len(items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        pools = {k: {g: v[:] for g, v in d.items()}
                 for k, d in (("real", self.real_by_gen), ("fake", self.fake_by_gen))}
        for d in pools.values():
            for v in d.values():
                rng.shuffle(v)

        half = self.batch_size // 2
        n_batches = self.n // self.batch_size
        for _ in range(n_batches):
            batch: List[int] = []
            gens = rng.sample(self.gens, min(len(self.gens), max(self.min_generators, 2)))
            per_gen = max(1, half // len(gens))
            for cls in ("real", "fake"):
                need = half
                for g in gens:
                    pool = pools[cls].get(g)
                    if not pool:
                        continue
                    take = min(per_gen, len(pool), need)
                    batch.extend(pool[:take])
                    del pool[:take]
                    need -= take
                # top up from any generator if a pool ran dry
                while need > 0:
                    avail = [g for g in self.gens if pools[cls].get(g)]
                    if not avail:
                        break
                    g = rng.choice(avail)
                    batch.append(pools[cls][g].pop())
                    need -= 1
            if len(batch) < self.batch_size and self.drop_last:
                break
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n // self.batch_size


def build_loader(
    items: Sequence[Tuple[str, int, str]],
    batch_size: int,
    train: bool = False,
    augment_cfg: Optional[dict] = None,
    num_workers: int = 4,
    balanced: bool = True,
    degradation: Optional[str] = None,
    seed: int = 42,
) -> DataLoader:
    aug = None
    if train and augment_cfg and augment_cfg.get("enabled", True):
        aug = RandomDegradation(
            jpeg_prob=augment_cfg.get("jpeg_prob", 0.5),
            jpeg_quality=tuple(augment_cfg.get("jpeg_quality", (50, 95))),
            blur_prob=augment_cfg.get("blur_prob", 0.3),
            blur_sigma=tuple(augment_cfg.get("blur_sigma", (0.5, 2.0))),
            resize_prob=augment_cfg.get("resize_prob", 0.3),
            resize_scale=tuple(augment_cfg.get("resize_scale", (0.5, 0.9))),
            color_jitter=augment_cfg.get("color_jitter", 0.1),
        )
    ds = GenImageDataset(
        items, train=train, augment=aug,
        hflip=(augment_cfg or {}).get("hflip", 0.5) if train else 0.0,
        degradation=degradation,
    )
    from ..utils.seed import worker_init_fn

    if train and balanced:
        sampler = BalancedGeneratorBatchSampler(items, batch_size, seed=seed)
        return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=True, drop_last=train, worker_init_fn=worker_init_fn)
