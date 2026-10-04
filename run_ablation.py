#!/usr/bin/env python
"""Run Heretic (Optuna TPE directional ablation) on google/gemma-2-2b-it on Apple Silicon.

Writes a config.toml, launches the `heretic` CLI, then verifies the export.

Heretic's last step is an interactive menu (no non-interactive flag exists in 1.4.0):
  1. "Which trial do you want to use?"  -> pick a Pareto-front trial (low refusals, low KL)
  2. "Save the model to a local folder" -> path: ./models/gemma-2-2b-it-abliterated
  (export strategy is preset to "merge", so you won't be asked about it)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

MODEL = "google/gemma-2-2b-it"
OUT_DIR = Path("./models/gemma-2-2b-it-abliterated")


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def write_config(path: Path, n_trials: int, n_startup: int, device: str, seed: int) -> None:
    # Dict device_map {"": device} pins the whole model to one device (no "auto" sharding).
    # bfloat16 first: Gemma 2 overflows in float16 (inf/NaN logits). float32 is the last resort.
    path.write_text(f'''\
model = "{MODEL}"
dtypes = ["bfloat16", "float32"]
quantization = "none"
device_map = {{ "" = "{device}" }}

n_trials = {n_trials}
n_startup_trials = {n_startup}   # random-exploration trials before TPE takes over
seed = {seed}

batch_size = 0                    # auto-probe
max_batch_size = 32               # keep the probe small on unified memory
max_response_length = 100
offload_outputs_to_cpu = true

export_strategy = "merge"         # full merged weights, loadable with plain from_pretrained
study_checkpoint_dir = "checkpoints"
''')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=25, help="total Optuna trials (default 25)")
    ap.add_argument("--startup-trials", type=int, default=10, help="random trials before TPE (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["mps", "cpu"], default=None, help="default: mps if available")
    ap.add_argument("--fresh", action="store_true", help="delete ./checkpoints to start a new study")
    args = ap.parse_args()

    if args.startup_trials >= args.trials:
        ap.error("--startup-trials must be smaller than --trials")
    if shutil.which("heretic") is None:
        sys.exit("`heretic` not on PATH. Activate the env: source heretic-env/bin/activate")
    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        sys.exit("No Hugging Face credentials found. Set HF_TOKEN or run `hf auth login` (Gemma 2 is gated).")

    device = args.device or pick_device()
    print(f"Device: {device}")
    if args.fresh:
        shutil.rmtree("checkpoints", ignore_errors=True)

    # Ops lacking an MPS kernel fall back to CPU instead of raising NotImplementedError.
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    # Heretic sets PYTORCH_ALLOC_CONF=expandable_segments:True (CUDA-only) when unset; pre-empt on MPS.
    env.setdefault("PYTORCH_ALLOC_CONF", "")

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    write_config(Path("config.toml"), args.trials, args.startup_trials, device, args.seed)
    print("Wrote config.toml. Launching Heretic...\n"
          f"When the menu appears: choose a trial -> 'Save the model to a local folder' -> {OUT_DIR}\n")

    rc = subprocess.call(["heretic", "--model", MODEL], env=env)  # inherits the TTY for the menu
    if rc != 0:
        print(f"heretic exited with code {rc}")
        return rc

    if list(OUT_DIR.glob("*.safetensors")) and (OUT_DIR / "config.json").exists():
        print(f"\nOK: abliterated model found at {OUT_DIR.resolve()}")
        return 0
    print(f"\nNo saved model at {OUT_DIR}. Re-run (the study resumes from ./checkpoints) and choose 'Save'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
