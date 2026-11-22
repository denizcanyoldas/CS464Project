"""Run the prespecified validation stages and resume from verified artifacts."""

import argparse
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoolake.__main__ import read_json, sha256


def read_toml(path):
    with path.open("rb") as source:
        return tomllib.load(source)


def stage_runs(stage, plan, defaults, device):
    options = {key: value for key, value in plan.items()
               if key not in {"models", "fractions", "augmentations", "seeds"}}
    if device is not None and options.get("device") != "cpu":
        options["device"] = device
    for model, seed, fraction, augment in product(
        plan["models"], plan["seeds"], plan["fractions"], plan["augmentations"],
    ):
        name = f"{stage}-{model}-f{fraction}-{augment}-s{seed}"
        yield name, {
            **defaults, **options, "model": model, "seed": seed,
            "fraction": fraction, "augment": augment,
        }


def complete(run, expected, split_hash):
    if not run.exists():
        return False
    metadata = read_json(run / "run.json")
    metrics = read_json(run / "metrics.json")
    if metadata["config"] != expected or metadata["split_sha256"] != split_hash:
        raise ValueError(f"Configuration differs for existing run {run}. Choose another output path.")
    if metrics["partition"] != "validation" or sha256(run / metadata["model_file"]) != metadata["model_sha256"]:
        raise ValueError(f"Existing run has inconsistent artifacts: {run}")
    return True


def cli_options(config):
    arguments = []
    for key, value in config.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            arguments.extend([f"--{flag}", str(value)])
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+")
    parser.add_argument("--study", type=Path, default=ROOT / "configs/final_study.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/runs/final-study")
    parser.add_argument("--split", type=Path, help="Use an existing frozen split manifest")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    config = ROOT / "configs/default.toml"
    defaults = read_toml(config)
    stages = read_toml(args.study)
    split = args.split.resolve() if args.split else ROOT / defaults["data"]["split_dir"] / "splits.json"
    digest = sha256(split)
    env = {
        **os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_HOME": str(ROOT / "artifacts/torch-cache"),
        "MPLCONFIGDIR": str(ROOT / "artifacts/matplotlib-cache"),
    }
    output = args.out.resolve()
    for stage in args.stages:
        if stage not in stages:
            parser.error(f"Unknown stage {stage}. Choose from {', '.join(stages)}.")
        for name, expected in stage_runs(stage, stages[stage], defaults["training"], args.device):
            run = output / name
            if complete(run, expected, digest):
                print(f"Verified existing validation run {name}", flush=True)
                continue
            print(f"Starting validation run {name}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "zoolake", "--config", str(config), "train",
                 "--split", str(split), "--out", str(run), *cli_options(expected)],
                cwd=ROOT, env=env, check=True,
            )


if __name__ == "__main__":
    main()
