# -*- coding: utf-8 -*-
"""
Config-driven runner for the module ablation study.

`ABLATION_DETAILS.md` specifies seven entries: the full pipeline plus six variants, each removing
exactly one module. This script turns that specification into an executable plan. Every variant is a
complete pipeline run into its own output subtree, and the only differences between them are which
stages are skipped, which checkpoint the next surviving stage initialises from, and the two
documented substitutions (static diagnosis for `no-T-GRPO`, a code-only reward for `no-HR`).

Nothing here reimplements a stage. Each step launches the ordinary stage script as a subprocess with
`ABLATION_*` environment variables set, and those variables are resolved by `ablation_config.py` at
the top of each script. So a stage run under this runner executes the same code path as a direct run,
and the plan below is the only place a variant's shape is described.

Usage (from `code/train`):

    python run_ablation.py --list                        # the variants and their stage plans
    python run_ablation.py --variant no-T-GRPO --dry-run # print the plan, run nothing
    python run_ablation.py --variant no-T-GRPO           # train it
    python run_ablation.py --all                         # train every variant
    python run_ablation.py --variant no-HR --evaluate    # score its final checkpoint

Notes on cost and resumption:

- Every variant trains from scratch by default. `--reuse-full` instead points each stage that is
  provably identical to the full pipeline's at the full run's existing checkpoint and skips
  retraining it, which is the optimisation `ABLATION_DETAILS.md` §11 describes. It is off by
  default because it silently couples a variant's result to a run that happened elsewhere.
- A step whose output already exists is skipped, so an interrupted run resumes. `--force` re-runs
  everything, `--only <stage>` re-runs from one stage onward.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ablation_config import env_name

# ==================== Layout ====================
TRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_DIR.parents[1]
MODEL_OUT = REPO_ROOT / "code" / "model" / "output"
BASE_MODEL = REPO_ROOT / "code" / "model" / "base" / "qwen2.5-14b-instruct"
ABLATION_MODEL_ROOT = MODEL_OUT / "ablation"
HARD_ROOT = REPO_ROOT / "data" / "train" / "Hard_samples" / "ablation"

# The RL corpus every variant shares. Stage 4 trains on it; stage 6 splits it 9:1 for validation.
RL_CONTENT_DATA = REPO_ROOT / "data" / "train" / "RL_data" / "GRPO_content" / "RL_content_samples.json"
COLD_START_DATA = REPO_ROOT / "data" / "train" / "RL_data" / "Cold_start" / "cold_start_samples.json"
FORMAT_DATA = REPO_ROOT / "data" / "train" / "RL_data" / "GRPO_format" / "RL_format_samples.json"

TXT2JSON = REPO_ROOT / "data" / "train" / "Hard_samples" / "txt2json.py"
AUGMENT_SCRIPT = TRAIN_DIR / "utils" / "Rules_augmented_hardsamples.py"
EVAL_SCRIPT = REPO_ROOT / "code" / "test" / "test.py"
EVAL_ROOT = REPO_ROOT / "code" / "test" / "eca_results"

# The name is load-bearing: stage 5 and stage 6 both read this file, and the augmentation script
# writes it under the same name.
AUGMENTED_FILENAME = "augmented_hard_samples.json"

# ==================== Stages ====================
STAGE_META = {
    "1dap": {"script": "1DAP.py", "out": "1DAP_output"},
    "2csft": {"script": "2cold_start_SFT.py", "out": "2Cold_start_output"},
    "3grpo": {"script": "3grpo_format.py", "out": "3GRPO_format_output"},
    "4grpo": {"script": "4grpo_content.py", "out": "4GRPO_content_output"},
    "5dap4hs": {"script": "5DAP_for_hardsamples.py", "out": "5DAP4HS_output"},
    # Stage 6 saves its converged artifact one level down, in `final/`
    "6it": {"script": "6model_iteration.py", "out": "6Iteration_output", "artifact": "final"},
}
STAGE_ORDER = ["1dap", "2csft", "3grpo", "4grpo", "5dap4hs", "6it"]


def checkpoint_dir(stage, out_dir: Path) -> Path:
    """Where a stage's saved model actually lands -- stage 6 nests it under `final/`."""
    artifact = STAGE_META[stage].get("artifact")
    return out_dir / artifact if artifact else out_dir


# ==================== Variant plans ====================
# Each entry is an ordered list of actions. `("stage", name, source)` means run that stage
# initialising from `source`, where source is "BASE" for the released base model or "@<stage>" for
# another stage's output. "hardsample_pipeline" expands to the convert-then-augment pair that turns
# stage 4's dump into the augmented set stage 5 trains on.
#
# `shareable` lists the stages whose inputs and settings are provably identical to the full
# pipeline's, so `--reuse-full` may take them from the full run instead of retraining. The lists are
# derived from the initialisation edges: a stage is shareable only while every variant that precedes
# it is untouched.
VARIANTS = {
    "full": {
        "removes": "-",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "2csft", "@1dap"),
            ("stage", "3grpo", "@2csft"),
            ("stage", "4grpo", "@3grpo"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@4grpo"),
            ("stage", "6it", "@5dap4hs"),
        ],
        "shareable": [],
        "reward_mode": "hybrid",
        "seed_augmented": True,
        "final": "6it",
    },
    "no-DAP": {
        "removes": "domain-adaptive pretraining (stage 1)",
        "plan": [
            ("stage", "2csft", "BASE"),
            ("stage", "3grpo", "@2csft"),
            ("stage", "4grpo", "@3grpo"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@4grpo"),
            ("stage", "6it", "@5dap4hs"),
        ],
        # Nothing is shared: every surviving stage starts from a different checkpoint than in `full`
        "shareable": [],
        "reward_mode": "hybrid",
        "seed_augmented": True,
        "final": "6it",
    },
    "no-CSFT": {
        "removes": "cold-start supervised fine-tuning (stage 2)",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "3grpo", "@1dap"),
            ("stage", "4grpo", "@3grpo"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@4grpo"),
            ("stage", "6it", "@5dap4hs"),
        ],
        # Removing the cold-start stage does not change how DAP is trained
        "shareable": ["1dap"],
        "reward_mode": "hybrid",
        "seed_augmented": True,
        "final": "6it",
    },
    "no-T-GRPO": {
        "removes": "both GRPO stages (3 and 4)",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "2csft", "@1dap"),
            # Substitute for the diagnosis the GRPO rollouts would have produced
            ("static_diagnose", "@2csft"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@2csft"),
            ("stage", "6it", "@5dap4hs"),
        ],
        "shareable": ["1dap", "2csft"],
        "reward_mode": "hybrid",
        "seed_augmented": True,
        "final": "6it",
    },
    "no-HSO": {
        "removes": "hard-sample diagnosis + 1:5 augmentation + targeted pretraining",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "2csft", "@1dap"),
            ("stage", "3grpo", "@2csft"),
            ("stage", "4grpo", "@3grpo"),
            # No conversion, no augmentation, no stage 5: the module is gone, not bypassed
            ("stage", "6it", "@4grpo"),
        ],
        "shareable": ["1dap", "2csft", "3grpo", "4grpo"],
        "reward_mode": "hybrid",
        # Stage 6 must not inherit the augmented set the removed module produced
        "seed_augmented": False,
        "final": "6it",
    },
    "no-HR": {
        "removes": "hybrid reward's judge term (stages 4 and 6)",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "2csft", "@1dap"),
            ("stage", "3grpo", "@2csft"),
            ("stage", "4grpo", "@3grpo"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@4grpo"),
            ("stage", "6it", "@5dap4hs"),
        ],
        # Stages 1-3 do not consume the reward HR lives in; the format reward in stage 3 is separate
        "shareable": ["1dap", "2csft", "3grpo"],
        "reward_mode": "code_only",
        "seed_augmented": True,
        "final": "6it",
    },
    "no-CIO": {
        "removes": "closed-loop iteration (stage 6)",
        "plan": [
            ("stage", "1dap", "BASE"),
            ("stage", "2csft", "@1dap"),
            ("stage", "3grpo", "@2csft"),
            ("stage", "4grpo", "@3grpo"),
            ("hardsample_pipeline",),
            ("stage", "5dap4hs", "@4grpo"),
        ],
        "shareable": ["1dap", "2csft", "3grpo", "4grpo", "5dap4hs"],
        "reward_mode": "hybrid",
        "seed_augmented": True,
        # The removed module is the last one, so this variant is scored at stage 5's checkpoint
        "final": "5dap4hs",
    },
}


# ==================== Plan construction ====================
class Step:
    """One executable action: a script, a working directory, and the environment it runs under."""

    def __init__(self, label, kind, script, cwd, env, done_check, note="", args=()):
        self.label = label
        self.kind = kind
        self.script = script
        self.cwd = cwd
        self.env = env
        self.done_check = done_check      # path that marks this step complete, or None
        self.note = note
        self.args = list(args)            # CLI args, for steps that are not stage scripts

    def is_done(self):
        if self.done_check is None:
            return False
        return (self.done_check / "config.json").exists()

    def env_for_subprocess(self):
        merged = dict(os.environ)
        merged.update({k: str(v) for k, v in self.env.items()})
        return merged

    def describe(self):
        return f"{self.label:<28} {self.kind:<17} {self.note}"


def variant_root(variant: str) -> Path:
    """
    Where a variant writes its stage outputs.

    `full` is the main pipeline, so it uses the pipeline's own output directory rather than a copy
    under `ablation/`. The other six are isolated under their own subtree so no variant can overwrite
    another's checkpoint or read it by accident.
    """
    return MODEL_OUT if variant == "full" else ABLATION_MODEL_ROOT / variant


def build_plan(variant: str, reuse_full: bool = False):
    """
    Resolve a variant's plan into concrete steps.

    Returns (steps, final_checkpoint). Stage directories are resolved as the plan is walked, so a
    stage taken from the full run automatically becomes the initialisation source for whatever
    follows it.
    """
    spec = VARIANTS[variant]
    root = variant_root(variant)
    full_root = variant_root("full")
    hard_dir = HARD_ROOT / variant

    steps = []
    stage_dirs = {}      # stage -> directory its checkpoint actually lives in
    sources = {}         # stage -> the directory that stage was initialised from

    def resolve(source):
        if source == "BASE":
            return BASE_MODEL
        if source.startswith("@"):
            return checkpoint_dir(source[1:], stage_dirs[source[1:]])
        raise ValueError(f"unrecognised source {source!r}")

    for action in spec["plan"]:
        kind = action[0]

        if kind == "stage":
            _, stage, source = action
            prev_dir = resolve(source)
            sources[stage] = prev_dir
            out_dir = root / STAGE_META[stage]["out"]

            shared = reuse_full and stage in spec["shareable"]
            if shared:
                full_dir = full_root / STAGE_META[stage]["out"]
                if checkpoint_dir(stage, full_dir).joinpath("config.json").exists():
                    stage_dirs[stage] = full_dir
                    steps.append(Step(
                        label=f"stage {stage}", kind="reuse", script=None, cwd=TRAIN_DIR, env={},
                        done_check=None, note=f"reuse full run: {full_dir}",
                    ))
                    continue

            env = {
                "ABLATION_VARIANT": variant,
                "ABLATION_REWARD_MODE": spec["reward_mode"],
                env_name(f"{stage}.MODEL_PATH"): prev_dir,
                env_name(f"{stage}.OUTPUT_DIR"): out_dir,
            }
            if stage == "2csft":
                env[env_name("2csft.DATA_PATH")] = COLD_START_DATA
            elif stage == "3grpo":
                env[env_name("3grpo.DATA_PATH")] = FORMAT_DATA
            elif stage == "4grpo":
                env[env_name("4grpo.DATA_PATH")] = RL_CONTENT_DATA
                env[env_name("4grpo.HARD_SAMPLES_TXT")] = hard_dir / "hard_samples.txt"
            elif stage == "5dap4hs":
                env[env_name("5dap4hs.AUGMENTED_SAMPLES_PATH")] = hard_dir / AUGMENTED_FILENAME
            elif stage == "6it":
                env[env_name("6it.BASE_DATA_PATH")] = RL_CONTENT_DATA
                env[env_name("6it.POOL_PATH")] = hard_dir / "iteration_pool.json"
                env[env_name("6it.AUGMENTED_PATH")] = hard_dir / "iteration_augmented.json"
                env[env_name("6it.AUGMENTED_SEED_PATH")] = hard_dir / AUGMENTED_FILENAME
                env["ABLATION_AUGMENTED_SEED_ENABLED"] = "1" if spec["seed_augmented"] else "0"

            stage_dirs[stage] = out_dir
            note = f"init from {prev_dir.name or prev_dir}"
            steps.append(Step(
                label=f"stage {stage}", kind="stage",
                script=TRAIN_DIR / STAGE_META[stage]["script"], cwd=TRAIN_DIR, env=env,
                done_check=checkpoint_dir(stage, out_dir), note=note,
            ))

        elif kind == "static_diagnose":
            _, source = action
            prev_dir = resolve(source)
            out_txt = hard_dir / "hard_samples.txt"
            env = {
                "ABLATION_VARIANT": variant,
                "ABLATION_REWARD_MODE": spec["reward_mode"],
                env_name("4grpo.HARD_SAMPLES_TXT"): out_txt,
            }
            steps.append(Step(
                label="static diagnosis", kind="static_diagnose",
                script=TRAIN_DIR / "ablation_static_diagnose.py", cwd=TRAIN_DIR, env=env,
                done_check=None,
                note=f"greedy over the RL split with {prev_dir.name or prev_dir}",
                args=["--model-path", str(prev_dir), "--data", str(RL_CONTENT_DATA),
                      "--out", str(out_txt)],
            ))

        elif kind == "hardsample_pipeline":
            env = {"ABLATION_VARIANT": variant}
            steps.append(Step(
                label="convert dump -> json", kind="txt2json", script=TXT2JSON, cwd=REPO_ROOT,
                env={**env,
                     env_name("txt2json.INPUT_TXT"): hard_dir / "hard_samples.txt",
                     env_name("txt2json.OUTPUT_JSON"): hard_dir / "origin_hard_samples.json"},
                done_check=None, note=f"-> {hard_dir.name}/origin_hard_samples.json",
            ))
            steps.append(Step(
                label="augment hard samples", kind="augment", script=AUGMENT_SCRIPT, cwd=TRAIN_DIR,
                env={**env,
                     env_name("augment.INPUT_JSON"): hard_dir / "origin_hard_samples.json",
                     env_name("augment.OUTPUT_JSON"): hard_dir / AUGMENTED_FILENAME},
                done_check=None, note="1:5, four error operators",
            ))

        else:
            raise ValueError(f"unknown action {kind!r} in variant {variant!r}")

    final_stage = spec["final"]
    if final_stage not in stage_dirs:
        raise ValueError(f"variant {variant!r} never produced its final stage {final_stage!r}")
    return steps, checkpoint_dir(final_stage, stage_dirs[final_stage])


# ==================== Execution ====================
def run_step(step, force=False, dry_run=False):
    if not force and step.is_done():
        print(f"  [skip] {step.label}: output already present ({step.done_check})")
        return True

    cmd = [sys.executable, str(step.script)] + step.args
    print(f"  [run ] {step.label} -> {step.script.name}  ({step.note})")
    if dry_run:
        for key in sorted(step.env):
            print(f"           {key}={step.env[key]}")
        if step.args:
            print(f"           argv={' '.join(step.args)}")
        return True

    result = subprocess.run(cmd, cwd=str(step.cwd), env=step.env_for_subprocess())
    if result.returncode != 0:
        print(f"  [FAIL] {step.label} exited with code {result.returncode}")
        return False
    return True


def evaluate(variant, dry_run=False):
    """Score a variant's final checkpoint with the shared evaluation harness."""
    _, final_dir = build_plan(variant)
    if not final_dir.exists():
        print(f"cannot evaluate {variant}: no checkpoint at {final_dir}")
        return False

    result_dir = EVAL_ROOT / variant
    env = {
        env_name("test.MODEL_PATH"): final_dir,
        env_name("test.RESULT_DIR"): result_dir,
    }
    print(f"\nEvaluate {variant}: {final_dir}")
    print(f"  results -> {result_dir}")
    if dry_run:
        for key in sorted(env):
            print(f"           {key}={env[key]}")
        return True

    merged = dict(os.environ)
    merged.update({k: str(v) for k, v in env.items()})
    result = subprocess.run([sys.executable, str(EVAL_SCRIPT)], cwd=str(EVAL_SCRIPT.parent),
                            env=merged)
    return result.returncode == 0


def show_variants():
    print("Variants (each removes exactly one module; `full` is the unmodified pipeline)\n")
    for name, spec in VARIANTS.items():
        stages = [a[1] for a in spec["plan"] if a[0] == "stage"]
        extras = [a[0] for a in spec["plan"] if a[0] in ("static_diagnose", "hardsample_pipeline")]
        print(f"  {name:<12} removes: {spec['removes']}")
        print(f"  {'':<12} stages: {' -> '.join(stages)}")
        if extras:
            print(f"  {'':<12} also  : {', '.join(extras)}")
        print(f"  {'':<12} reward: {spec['reward_mode']}, "
              f"final checkpoint: {spec['final']}, "
              f"reusable stages: {spec['shareable'] or 'none'}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the module ablation study.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="variant to run")
    parser.add_argument("--all", action="store_true", help="run every variant")
    parser.add_argument("--list", action="store_true", help="show the variants and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plans, run nothing")
    parser.add_argument("--evaluate", action="store_true",
                        help="score the variant's final checkpoint instead of training")
    parser.add_argument("--force", action="store_true", help="re-run steps whose output exists")
    parser.add_argument("--reuse-full", action="store_true",
                        help="take provably-identical stages from the full run instead of retraining")
    args = parser.parse_args(argv)

    if args.list:
        show_variants()
        return 0

    variants = sorted(VARIANTS) if args.all else ([args.variant] if args.variant else [])
    if not variants:
        parser.error("pass --variant, --all, or --list")
    if args.all and args.evaluate:
        parser.error("--all with --evaluate would score six checkpoints in sequence; "
                     "run them one at a time so a failure is attributable")

    print(f"repo root : {REPO_ROOT}")
    print(f"variants  : {', '.join(variants)}")
    print(f"reuse full: {args.reuse_full}")
    print()

    failures = []
    for variant in variants:
        steps, final_dir = build_plan(variant, reuse_full=args.reuse_full)
        print("=" * 78)
        print(f"VARIANT {variant} -- removes {VARIANTS[variant]['removes']}")
        print(f"final checkpoint: {final_dir}")
        print("=" * 78)

        if args.evaluate:
            if not evaluate(variant, dry_run=args.dry_run):
                failures.append(f"{variant}: evaluation")
            continue

        print(f"{'STEP':<28} {'KIND':<17} NOTE")
        for step in steps:
            print(step.describe())
        print()

        for step in steps:
            if not run_step(step, force=args.force, dry_run=args.dry_run):
                # A failed stage invalidates everything downstream of it, so stop this variant
                # rather than launching the next stage against a checkpoint that was never written.
                failures.append(f"{variant}: {step.label}")
                break
        else:
            print(f"\n{variant} complete -> {final_dir}")

    print()
    if failures:
        print(f"{len(failures)} step(s) failed or were not run:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all requested work finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
