"""Clean (fault-free) baseline: generate images from google/ddpm-cifar10-32 with a DDIM sampler.

Every setting is a command-line argument and is written to <out>/run.json together with the
software/hardware environment and a hash of the raw output, so a run can be reproduced and
compared exactly (see compare_runs.py). No fault injection happens here.

Example:
    python src/baseline.py --seed 0 --steps 50 --run-name seed0_a
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMPipeline
from huggingface_hub import HfApi
from PIL import Image

PACKAGES = ["torch", "diffusers", "transformers", "accelerate", "safetensors",
            "huggingface_hub", "numpy", "pillow"]


def run_cmd(cmd):
    """Return stripped stdout of a command, or None if it is unavailable or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def os_description():
    desc = {"platform": platform.platform(), "release": platform.release()}
    if sys.platform.startswith("linux"):
        desc["wsl"] = "microsoft" in platform.release().lower()
        try:
            info = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
            desc["distro"] = info.get("PRETTY_NAME", "").strip('"')
        except OSError:
            pass
    return desc


def environment(device):
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    env = {
        "python": platform.python_version(),
        "os": os_description(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "git_commit": commit,
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain"])) if commit else None,
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["nvidia_driver"] = run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return env


def save_grid(images, path):
    w, h = images[0].size
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    grid.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/ddpm-cifar10-32")
    ap.add_argument("--revision", default="main", help="Hub branch/tag/commit; the resolved commit is logged")
    ap.add_argument("--seed", type=int, default=0, help="Image-noise seed (pilot placeholder, not a protocol choice)")
    ap.add_argument("--num-images", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50, help="DDIM num_inference_steps (pilot placeholder)")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM eta; 0.0 = deterministic sampler")
    ap.add_argument("--device", default="cuda", help="cuda (default; fails loudly if unavailable) or cpu")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                         "(only if two default runs differ; record that it was needed)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", default="logs")
    args = ap.parse_args()

    if args.deterministic:
        # Must happen before any CUDA work in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False; pass --device cpu to run on CPU deliberately.")

    run_name = args.run_name or f"seed{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out_root) / run_name
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    (out / "images").mkdir()

    revision_sha = HfApi().model_info(args.model, revision=args.revision).sha
    pipe = DDIMPipeline.from_pretrained(args.model, revision=revision_sha).to(device)

    # CPU generator: the initial noise is then identical regardless of the device that runs the UNet.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    samples = pipe(batch_size=args.num_images, generator=generator, num_inference_steps=args.steps,
                   eta=args.eta, output_type="np").images  # float32 (N, 32, 32, 3) in [0, 1]
    elapsed = time.perf_counter() - t0

    np.save(out / "samples.npy", samples)
    images = pipe.numpy_to_pil(samples)
    for i, img in enumerate(images):
        img.save(out / "images" / f"{i:04d}.png")
    save_grid(images, out / "grid.png")

    results = {
        "samples_sha256": hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest(),
        "samples_shape": list(samples.shape),
        "all_finite": bool(np.isfinite(samples).all()),
        "generation_seconds": round(elapsed, 3),
    }
    if device.type == "cuda":
        results["peak_gpu_mem_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        results["peak_gpu_mem_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)
    if sys.platform != "win32":
        import resource
        # ru_maxrss is KiB on Linux, bytes on macOS.
        scale = 2**20 if sys.platform == "darwin" else 2**10
        results["peak_process_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable] + sys.argv,
        "settings": {
            "model": args.model,
            "revision_requested": args.revision,
            "revision_sha": revision_sha,
            "seed": args.seed,
            "generator_device": "cpu",
            "num_images": args.num_images,
            "num_inference_steps": args.steps,
            "eta": args.eta,
            "deterministic_algorithms": args.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "unet_dtype": str(pipe.unet.dtype),
            "unet_params": sum(p.numel() for p in pipe.unet.parameters()),
            "scheduler_class": type(pipe.scheduler).__name__,
            # _use_default_values is built from a set; sort it so identical runs log identically.
            "scheduler_config": {k: sorted(v) if k == "_use_default_values" else v
                                 for k, v in pipe.scheduler.config.items()},
            "timesteps_used": pipe.scheduler.timesteps.tolist(),
        },
        "environment": environment(device),
        "results": results,
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"Saved {len(images)} images to {out}  sha256={results['samples_sha256'][:16]}  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
