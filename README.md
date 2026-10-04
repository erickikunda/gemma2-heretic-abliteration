# gemma2-heretic-abliteration

Heretic abliteration of `google/gemma-2-2b-it` on Apple Silicon (MPS).

> Wrapper scripts around [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0) by Philipp Emanuel Weidmann. Not affiliated with the Heretic project or Google. Gemma is subject to Google's [Gemma terms](https://ai.google.dev/gemma/terms), which also apply to derivative weights.

Files: `setup.sh` (env) · `run_ablation.py` (Heretic wrapper) · `evaluate.py` (before/after comparison).
Tested against `heretic-llm` 1.4.0's real config schema. The full end-to-end run (model download + trials) needs your HF token, so it has not been executed here.

## 0. Prerequisites
- macOS on Apple Silicon, [`uv`](https://docs.astral.sh/uv/) installed. (No conda needed; uv provisions Python 3.12.
  Heretic 1.4.0 declares support for 3.10–3.12, so 3.12 is safer than a newer system Python.)
- A Hugging Face account that has **accepted the Gemma license** at https://huggingface.co/google/gemma-2-2b-it
  and a read token.

## 1. Environment + auth
```bash
export HF_TOKEN=hf_xxxxxxxx      # or: run `hf auth login` after step 1
bash setup.sh                    # creates ./heretic-env, installs deps, checks MPS + HF access
source heretic-env/bin/activate
```
`setup.sh` fails loudly if you're unauthenticated or haven't accepted the license.
It installs torch/torchvision/torchaudio, transformers, accelerate, optuna, datasets, heretic-llm.

> **bitsandbytes:** `heretic-llm` lists it as a hard dependency and `heretic/model.py` imports it at module load,
> so it gets installed transitively. It is never used here (`quantization = "none"`), and nothing calls CUDA.

## 2. Run the ablation
```bash
python run_ablation.py                 # 25 trials, 10 random startup, then Optuna TPE
python run_ablation.py --trials 30 --startup-trials 12
python run_ablation.py --fresh         # discard ./checkpoints and start a new study
```
The wrapper writes `config.toml` (bf16 → fp32 fallback, whole model pinned to `mps`, `export_strategy = "merge"`)
and launches `heretic`. The study is checkpointed to `./checkpoints`, so Ctrl-C and re-running resumes it.

Heretic 1.4.0 has **no non-interactive export flag**, so when the trials finish you answer its menu:
1. *Which trial do you want to use?* → pick a Pareto-front entry (few refusals, low KL divergence; the first listed is usually the best trade-off).
2. *Save the model to a local folder* → `./models/gemma-2-2b-it-abliterated`
3. *Exit* / return to menu.

The wrapper then verifies `*.safetensors` + `config.json` exist in that folder.

## 3. Evaluate
```bash
python evaluate.py
```
Loads baseline, generates 10 greedy completions (`do_sample=False`, which is HF's way of saying temperature 0; `temperature=0.0` itself raises an error), frees it, then does the same with the abliterated model.
Prints each pair, then a summary table with refusal check, coherence heuristic, and, for the 5 benign prompts, a task-specific correctness check (time arithmetic, function defined, JSON keys, exactly 3 list items, syllogism).
Full outputs go to `results/report.md` and `results/report.json`.

Expect: refusals on the boundary set drop vs. baseline, while benign correctness and coherence stay about the same.
Caveats: Gemma-2-2b-it refuses little on these mild prompts to begin with, so the baseline may already score low; the refusal check is substring-based (can false-positive on a story that contains "I'm sorry"); the coherence check is a heuristic, not a quality metric. For a real regression check, use Heretic's built-in "Benchmark the model" menu action (lm-eval).

## Troubleshooting (MPS)
Note on "custom residual hooks": Heretic 1.4.0 does not register forward hooks. It reads residuals through `output_hidden_states` and applies ablation as rank-1/low-rank **LoRA adapters via PEFT**, merged on export. The MPS issues below come from that path.

| Symptom | Cause / fix |
|---|---|
| `NotImplementedError: ... not currently implemented for the MPS device` | Op lacks an MPS kernel. The wrapper sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (CPU fallback, slower). Set it yourself if running `heretic` directly. |
| `inf`/`nan` probability errors, garbage text | Gemma 2 overflows in **float16**. `config.toml` tries `bfloat16` first; don't add float16. bf16 on MPS needs a recent macOS (14+) and torch. |
| Heretic prints dtype "Failed" then falls through to float32 | Expected fallback; float32 (~10 GB) fits in 24 GB but is slower. |
| `RuntimeError: MPS backend out of memory` / system swaps | Lower `max_batch_size` in `config.toml` (try 8), close other apps, or `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to remove the allocator cap (can freeze the Mac if you overshoot). |
| Tensors on `cpu` and `mps` mix (`Expected all tensors to be on the same device`) | Caused by `device_map = "auto"` splitting/offloading. The wrapper pins `{ "" = "mps" }`. If a CPU-fallback op triggers it, re-run with `--device cpu`. |
| `UserWarning: expandable_segments not supported` | Heretic sets a CUDA-only allocator flag by default; harmless on MPS (wrapper blanks it). |
| `ImportError: bitsandbytes` / libbitsandbytes load errors | It's imported but unused. `uv pip install -U bitsandbytes`; if no wheel for your platform, create a stub: put an empty `bitsandbytes/__init__.py` plus `BitsAndBytesConfig` is from transformers, so an empty module is enough for the unquantized path. |
| `GatedRepoError` / 401 | Token missing, or Gemma license not accepted on the model page. |
| Results differ run to run | MPS kernels aren't bit-deterministic; trial scores vary slightly even with `seed`. Greedy eval is stable enough for coarse comparison. |
| Trial speed | Expect minutes per trial on an M-series GPU; 25 trials is roughly 30–90 min. This is an estimate, not a measurement. |
