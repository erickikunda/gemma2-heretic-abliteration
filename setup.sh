#!/usr/bin/env bash
# Creates ./heretic-env (uv-managed venv) and installs everything needed.
# Usage: bash setup.sh
set -euo pipefail

ENV_DIR="heretic-env"
PY_VERSION="3.12"   # heretic-llm 1.4.0 officially supports 3.10-3.12; 3.14 is untested.

command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || echo "WARNING: not Apple Silicon; MPS will be unavailable."

uv venv "$ENV_DIR" --python "$PY_VERSION"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# macOS arm64 wheels from PyPI include the MPS backend; no extra index needed.
uv pip install torch torchvision torchaudio
uv pip install transformers accelerate optuna datasets huggingface_hub
# NOTE: heretic-llm hard-depends on bitsandbytes and imports it unconditionally
# (heretic/model.py). It is installed as a transitive dependency but never used:
# we run with quantization = "none". There is no CUDA code path exercised.
uv pip install heretic-llm

echo "--- sanity checks ---"
python - <<'PY'
import torch, transformers, optuna
print("torch", torch.__version__, "| transformers", transformers.__version__, "| optuna", optuna.__version__)
print("MPS built:", torch.backends.mps.is_built(), "| MPS available:", torch.backends.mps.is_available())
try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes imports OK (unused)")
except Exception as e:
    print("bitsandbytes import FAILED:", e, "\n -> see README troubleshooting")
PY

echo "--- Hugging Face auth (Gemma 2 is gated) ---"
python - <<'PY'
import os, sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
try:
    who = HfApi().whoami()["name"]
    print("Authenticated as:", who)
except Exception:
    print("NOT authenticated. Do ONE of:\n  export HF_TOKEN=hf_xxx   (read-scope token)\n  hf auth login")
    sys.exit(1)
try:
    hf_hub_download("google/gemma-2-2b-it", "config.json")
    print("Access to google/gemma-2-2b-it: OK")
except GatedRepoError:
    print("Token is valid but you have not accepted the Gemma license.\n"
          "Open https://huggingface.co/google/gemma-2-2b-it, click 'Acknowledge license', then re-run.")
    sys.exit(1)
except HfHubHTTPError as e:
    print("HF error:", e); sys.exit(1)
PY
echo "Setup complete. Activate with: source $ENV_DIR/bin/activate"
