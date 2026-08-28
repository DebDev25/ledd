import os

from ledd.utils.config import apply_overrides, load_config

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def test_default_config_loads():
    cfg = load_config(os.path.join(CFG, "default.yaml"))
    assert cfg["model"]["frequency"]["n_bands"] == 12
    assert cfg["train"]["select_metric"] == "val_generator_auc"


def test_base_inheritance_overrides_only_specified_keys():
    cfg = load_config(os.path.join(CFG, "stage2_frequency.yaml"))
    assert cfg["stage"] == "frequency"
    assert cfg["train"]["batch_size"] == 256           # overridden
    assert cfg["model"]["frequency"]["n_bands"] == 12  # inherited


def test_ablation_configs_inherit_through_two_levels():
    cfg = load_config(os.path.join(CFG, "ablations", "fusion_concat.yaml"))
    assert cfg["model"]["fusion"]["mode"] == "concat"
    assert cfg["stage"] == "joint"
    assert cfg["data"]["image_size"] == 224


def test_generator_roles_are_disjoint_in_shipped_config():
    cfg = load_config(os.path.join(CFG, "default.yaml"))
    d = cfg["data"]
    roles = list(d["train_generators"]) + [d["val_generator"]] + list(d["test_generators"])
    assert len(roles) == len(set(roles)), "shipped config leaks a generator across roles"


def test_cli_overrides():
    cfg = apply_overrides(load_config(os.path.join(CFG, "default.yaml")),
                          ["train.lr_new=0.5", "model.frequency.n_bands=16"])
    assert cfg["train"]["lr_new"] == 0.5
    assert cfg["model"]["frequency"]["n_bands"] == 16
