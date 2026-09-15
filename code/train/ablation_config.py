# -*- coding: utf-8 -*-
"""
Ablation variant configuration layer.

Every stage script declares its paths and switches through `override(key, default)`. When no
matching environment variable is set the default wins, so running a stage script directly behaves
exactly as it did before this module existed. `run_ablation.py` drives the variants by exporting
these variables before launching each stage.

Environment variable naming:  ABLATION_<KEY>, with `.` replaced by `_` and the key upper-cased.
For example `override("4grpo.MODEL_PATH", ...)` reads `ABLATION_4GRPO_MODEL_PATH`.

Keeping the override in the environment rather than in an importable settings object is deliberate:
the stage scripts are launched as subprocesses, and several of them execute training at import time,
so there is no clean way to hand them a config object from the parent process.
"""

import os

PREFIX = "ABLATION_"


def env_name(key):
    """
    The environment variable a given key reads from.

    Exposed so a launcher can construct the names it needs to set without restating the naming rule,
    which is the kind of duplication that silently stops matching after one side is edited.
    """
    return PREFIX + key.upper().replace(".", "_")


def override(key, default):
    """
    Return the environment override for `key`, or `default` when unset.

    An empty string counts as unset: it is far more often a shell mistake (`VAR= python x.py`)
    than a deliberate instruction to use a blank path.
    """
    value = os.environ.get(env_name(key))
    return value if value else default


def variant():
    """Name of the running ablation variant, or None when this is a normal pipeline run."""
    return override("VARIANT", "") or None


def reward_mode(default="hybrid"):
    """
    Training reward variant.

    "hybrid"    -- code reward and LLM-as-a-judge combined at 0.1 : 0.9 (paper Eq. 7).
    "code_only" -- the code reward alone, on its native 0-10 scale, with no judge call. Used by
                   the `no-HR` ablation.

    The code-only branch deliberately keeps the 0-10 scale instead of rescaling to 0-1. The
    hard-sample threshold is defined in absolute reward units, so rescaling would change which
    samples are diagnosed as hard at the same time as it removes the judge -- two variables at
    once. Keeping the scale fixed keeps the pool's membership criterion comparable across variants.
    """
    return override("REWARD_MODE", default)


def augmented_seed_enabled(default=True):
    """
    Whether stage 6 seeds its accumulated augmented set from the augmented hard-sample file.

    Set to "0" by the `no-HSO` variant, where the hard-sample module is removed and the loop must
    start from an empty pool with nothing inherited. Without this switch the loop would still pick
    up the augmented file the removed module produced and silently import part of that module's
    effect back in.
    """
    return override("AUGMENTED_SEED_ENABLED", "1" if default else "0") != "0"


def announce(stage: str):
    """Print the active variant so a multi-stage log can be read back per variant."""
    name = variant()
    if name:
        print(f"[ablation] variant={name} stage={stage} reward_mode={reward_mode()}")
    return name
