# Bit-Flip Faults in Diffusion Models (Tier S: CIFAR-10 DDPM)

AI Collective R&D Lab, UC Davis (Digital System Design / Test / Verification + Fault Tolerance Research Group, Prof. Hussain Al-Asaad). Mentor: Parnian Shabani Kamran. Team: Tony Lai, Aneesh.

We study silent data corruption from hardware bit-flip faults in diffusion image generators. A flipped bit in a stored weight can produce either an obviously broken image (easy to catch) or a realistic image whose content changed (silent corruption, the dangerous case).

> **Status:** environment and clean baseline only. The fault-injection procedure is being standardized by Parnian; no injector code exists yet by design.

## Research questions

- **RQ1 (required).** Under randomly chosen persistent single-bit FP16 weight faults, how do the text encoder, UNet blocks and VAE decoder differ in their rates of human-validated semantic change and visual corruption?
- **RQ2 (extension).** With the same model and sampler, do more denoising steps reduce the effect of a transient fault injected at the same noise level and size?
- **RQ3 (scale check).** Do conclusions from a small model hold for full Stable Diffusion 1.5?

Expected, not assumed: exponent bit 14 dominates everywhere (H1); decoder faults give corruption more often than meaning change (H2); more remaining steps reduce transient damage (H3).

Tier S model: [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) (35.7M params, FP32 checkpoint), sampled with DDIM. It has no text encoder, so a pretrained classifier's label change stands in for "meaning change".

## Layout

```
src/        baseline.py (clean generation + run record), compare_runs.py (bitwise check)
configs/    (empty until the standardized fault procedure lands)
logs/       one folder per run: run.json (committed) + images/, grid.png, samples.npy (git-ignored)
notebooks/
```

## Setup (results machine: RTX 5090)

All results must come from one GPU, one OS environment and one pinned software stack.

1. Check the driver: `nvidia-smi`. The "CUDA Version" it reports must be at least the CUDA version of the wheel you install.
2. Python 3.11+ (3.12 or 3.13 have Blackwell wheels). Create a virtual environment:
   - Linux / WSL: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `py -3.12 -m venv .venv` then `.venv\Scripts\activate`
3. Install PyTorch with CUDA >= 12.8 (Blackwell needs torch >= 2.7). Confirm the command at https://pytorch.org/get-started/locally/. As of Nov 3, 2026, `torch 2.15.0+cu134` exists for both Windows and Linux:
   `pip install torch --index-url https://download.pytorch.org/whl/cu134`
   If the driver is too old for CUDA 13.4, update the driver or use the `cu128` index (torch 2.11.0).
4. `pip install diffusers transformers accelerate safetensors pillow numpy`
5. Smoke test: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `True` and `NVIDIA GeForce RTX 5090`.
6. Pin the stack: `pip freeze > requirements.txt`

To recreate the pinned stack later (new venv or another clone), install from `requirements.txt`. The extra index is needed because the `+cu134` torch build is not on PyPI:
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu134`

## Clean baseline + determinism check

Run from the repo root. Each run is its own process; seed and step count are pilot placeholders, not protocol decisions.

```
python src/baseline.py --seed 0 --steps 50 --run-name seed0_a
python src/baseline.py --seed 0 --steps 50 --run-name seed0_b
python src/compare_runs.py logs/seed0_a logs/seed0_b > logs/seed0_compare.json
```

`compare_runs.py` exits 0 only if the settings match and the raw float outputs are bitwise identical. If they differ, record that as a finding, then repeat both runs with `--deterministic` (sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`). If torch then raises an error naming an op with no deterministic implementation, record that too.

Each `run.json` records: model id and resolved Hub commit, seed, sampler class and full scheduler config, step count, eta, the timesteps actually used, library versions, OS, GPU, driver, git commit (and whether the tree was dirty), SHA-256 of the raw output, wall time and peak GPU memory.

Notes:
- The checkpoint ships with a `DDPMScheduler` (its training scheduler). `DDIMPipeline` converts it to a `DDIMScheduler` with the same training noise schedule (linear betas 1e-4 to 0.02, 1000 train timesteps); `run.json` logs the resulting config.
- The image-noise generator is a CPU `torch.Generator`, so the initial noise for a given seed is the same on any device.
- `--device cuda` is the default and fails loudly if CUDA is unavailable, so a broken install can't silently fall back to CPU.

## Open decisions (set by Parnian's standardized procedure; not baked into code)

Fault target (weights / input noise / activations), single-bit vs multi-bit model, FP16 cast vs FP32 bit numbering, sampler and step count for the campaign, and how output change is evaluated.
