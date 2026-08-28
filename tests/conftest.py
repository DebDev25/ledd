"""Shared fixtures.

Note: torch is NOT imported at module level. A module-level `importorskip` in
conftest skips the ENTIRE test directory, which would silently hide the
config/data tests that need no torch at all.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tiny_cfg():
    """Small config so the model suite runs on CPU in seconds."""
    return {
        "seed": 0,
        "data": {"image_size": 64},
        "model": {
            "spatial": {"name": "mobilevit_s", "pretrained": False, "token_stage": 3},
            "frequency": {"n_bands": 8, "drop_lowest": 1, "representation": "magnitude",
                          "embed_dim": 32, "depth": 2, "heads": 4, "use_radial_mlp": True},
            "fusion": {"mode": "cross_attention", "dim": 32, "heads": 4, "layers": 1,
                       "bidirectional": True, "modality_dropout": 0.25},
            "head": {"hidden": 32, "dropout": 0.0},
            "projection": {"dim": 16, "hidden": 32},
        },
        "loss": {
            "bce_weight": 1.0, "supcon_weight": 0.5, "band_entropy_weight": 0.05,
            "supcon": {"temperature": 0.1, "queue_size": 64, "warmup_steps": 0, "ramp_steps": 1},
            "band_entropy": {"normalize": True},
        },
    }
