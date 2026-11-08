"""MNIST bit-flip study (plan-mnist-week1.md, steps 1-6). Run from mnist/:

    python src/mnist_fault.py train --run-name train_a
    python src/mnist_fault.py one-weight --model logs/train_a/model_fp32.pt --run-name one_weight
    python src/mnist_fault.py campaign --model logs/train_a/model_fp32.pt --run-name campaign_a
    python src/mnist_fault.py summarize logs/campaign_a
    python src/mnist_fault.py compare logs/campaign_a logs/campaign_b

Settings come from configs/mnist.json. Each run writes logs/<run_name>/ and refuses to overwrite it.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitflip import field, fingerprint, flip, restore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # mnist/
CFG = json.loads((ROOT / "configs" / "mnist.json").read_text())
RAW_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
CLASSES = ["masked", "sdc", "non_finite"]


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def seed_sequence(role_name):
    return np.random.SeedSequence(CFG["master_seed"], spawn_key=(CFG["roles"][role_name],))


def derived_seeds():
    """Every role's seed as int(SeedSequence.generate_state(1)[0]); torch uses it, numpy uses the SeedSequence."""
    return {name: int(seed_sequence(name).generate_state(1)[0]) for name in CFG["roles"]}


def rng(role_name):
    return np.random.default_rng(seed_sequence(role_name))


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device):
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    return {
        "command": shlex.join(sys.argv),
        "git_commit": commit,
        # Same convention as ../cifar10/src/baseline.py: untracked files (earlier logs/) don't count.
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain", "--untracked-files=no"])) if commit else None,
        "master_seed": CFG["master_seed"],
        "roles": CFG["roles"],
        "derived_seeds": derived_seeds(),
        "device": device,
        "macos": platform.mac_ver()[0],
        "chip": run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ["torch", "torchvision", "numpy"]},
        "torch_num_threads": torch.get_num_threads(),
    }


def new_run_dir(run_name):
    out = ROOT / "logs" / run_name
    out.parent.mkdir(exist_ok=True)
    out.mkdir()  # raises FileExistsError: never overwrite a run
    return out


def write_run_json(out, info, t0):
    info["wall_time_s"] = round(time.perf_counter() - t0, 3)
    info["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")


def load_mnist():
    """Normalized (N, 784) float32 tensors on CPU, plus SHA-256 of the four uncompressed raw files."""
    d = CFG["data"]
    root = ROOT / d["root"]
    out = []
    for train in (True, False):
        ds = torchvision.datasets.MNIST(root=root, train=train, download=True)
        x = ((ds.data.float() / 255 - d["normalize_mean"]) / d["normalize_std"]).reshape(-1, 784)
        out += [x, ds.targets]
    hashes = {f: sha256_file(root / "MNIST" / "raw" / f) for f in RAW_FILES}
    return (*out, hashes)


def build_model():
    m = CFG["model"]
    require(m["activation"] == "relu", f"unsupported activation {m['activation']}")
    n_in, n_hidden, n_out = m["layers"]
    model = torch.nn.Sequential(OrderedDict(
        fc1=torch.nn.Linear(n_in, n_hidden), relu=torch.nn.ReLU(), fc2=torch.nn.Linear(n_hidden, n_out)))
    n_params = sum(p.numel() for p in model.parameters())
    n_bias = sum(p.numel() for n, p in model.named_parameters() if n.endswith("bias"))
    require(n_params == 101_770 and n_bias == 138, f"parameter count {n_params}, bias {n_bias}")
    return model


def param_hash(model):
    h = hashlib.sha256()
    for t in model.state_dict().values():
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, x):
    """Logits for all test images, copied to CPU (this also synchronizes MPS)."""
    return model(x).cpu()


def correct_count(logits, y):
    return int((logits.argmax(dim=1) == y).sum())


# ---------------------------------------------------------------- step 1

def cmd_train(args):
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    tc = CFG["train"]
    require((tc["device"], tc["dtype"], tc["optimizer"], tc["loss"]) == ("cpu", "float32", "adam", "cross_entropy"),
            f"unsupported train settings {tc}")
    x_train, y_train, x_test, y_test, mnist_hashes = load_mnist()
    seed = derived_seeds()["train"]

    torch.manual_seed(seed)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    n, bs = x_train.shape[0], tc["batch_size"]
    t_train = time.perf_counter()
    for epoch in range(tc["epochs"]):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}: mean loss {total / n:.4f}")
    train_wall = time.perf_counter() - t_train

    model.eval()
    correct = correct_count(evaluate(model, x_test), y_test)
    torch.save(model.state_dict(), out / "model_fp32.pt")
    info = environment(tc["device"])
    info.update({
        "mnist_raw_sha256": mnist_hashes,
        "model_file_sha256": sha256_file(out / "model_fp32.pt"),
        "model_param_sha256": param_hash(model),
        "fp32_cpu_accuracy": correct / len(y_test),
        "train_wall_time_s": round(train_wall, 3),
    })
    write_run_json(out, info, t0)
    print(f"fp32_cpu_accuracy {info['fp32_cpu_accuracy']:.4f}  param hash {info['model_param_sha256']}")


# ---------------------------------------------------------------- steps 2 and 3

def classify(logits, ref_preds, y):
    """Returns (class, correct). Order: non_finite, then sdc, then masked."""
    if not torch.isfinite(logits).all():
        cls = "non_finite"
    elif (logits.argmax(dim=1) != ref_preds).any():
        cls = "sdc"
    else:
        cls = "masked"
    return cls, correct_count(logits, y)


def inject(m16, name, index, bit, x16, y, ref_preds):
    """One fault: fingerprint, flip, evaluate, classify, restore, verify. Returns (old, new, class, correct)."""
    p = dict(m16.named_parameters())[name]
    fp = fingerprint(p)
    old = p.data.view(-1)[index].item()
    pattern = flip(m16, name, index, bit)
    try:
        new = p.data.view(-1)[index].item()
        cls, correct = classify(evaluate(m16, x16), ref_preds, y)
    finally:
        restore(m16, name, index, pattern)
    require(fingerprint(p) == fp, f"fingerprint mismatch after restoring {name}[{index}] bit {bit}")
    return old, new, cls, correct


def prepare(model_path):
    """Steps 2-3: FP32 and FP16 copies on MPS, margin check, double-eval check, reference, sanity check."""
    ec = CFG["eval"]
    require(ec["dtype"] == "float16", f"unsupported eval dtype {ec['dtype']}")
    dev = ec["device"]
    _, _, x_test, y_test, mnist_hashes = load_mnist()
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    train_info = json.loads((Path(model_path).parent / "run.json").read_text())
    require(param_hash(model) == train_info["model_param_sha256"], "model does not match its training run.json")
    model.eval()
    m32 = copy.deepcopy(model).to(dev)
    m16 = copy.deepcopy(model).half().to(dev)
    x32, x16 = x_test.to(dev), x_test.half().to(dev)
    n = len(y_test)

    c32 = correct_count(evaluate(m32, x32), y_test)
    ref_logits = evaluate(m16, x16)
    c16 = correct_count(ref_logits, y_test)
    diff_pp = abs(c16 - c32) * 100 / n
    require(diff_pp <= ec["fp16_margin_pp"],
            f"FP16 margin failed: fp16 {c16}/{n}, fp32 {c32}/{n}, diff {diff_pp} pp > {ec['fp16_margin_pp']}")
    again = evaluate(m16, x16)
    logits_sha = sha256_bytes(ref_logits.numpy().tobytes())
    again_sha = sha256_bytes(again.numpy().tobytes())
    require(logits_sha == again_sha, f"clean FP16 logits differ between two evaluations: {logits_sha} vs {again_sha}")
    ref_preds = ref_logits.argmax(dim=1)

    # Sanity check (no RNG): bit 14 of the fc1.weight element with the largest FP16 magnitude below 1.
    w = m16.fc1.weight.detach().cpu().view(-1).abs()
    idx = int(torch.where(w < 1, w, torch.full_like(w, -1)).argmax())
    fp = fingerprint(m16.fc1.weight)
    old = m16.fc1.weight.data.view(-1)[idx].item()
    pattern = flip(m16, "fc1.weight", idx, 14)
    try:
        new = m16.fc1.weight.data.view(-1)[idx].item()
        flipped = evaluate(m16, x16)
    finally:
        restore(m16, "fc1.weight", idx, pattern)
    restored = evaluate(m16, x16)
    sanity = {
        "tensor": "fc1.weight", "element_index": idx, "bit": 14,
        "original_value": repr(old), "new_value": repr(new),
        "logits_changed": not torch.equal(flipped, ref_logits),
        "predictions_changed": int((flipped.argmax(dim=1) != ref_preds).sum()),
        "logits_finite": bool(torch.isfinite(flipped).all()),
        "accuracy_flipped": correct_count(flipped, y_test) / n,
        "restored_logits_match": sha256_bytes(restored.numpy().tobytes()) == logits_sha,
        "restored_fingerprint_match": fingerprint(m16.fc1.weight) == fp,
    }
    require(sanity["logits_changed"] and sanity["restored_logits_match"] and sanity["restored_fingerprint_match"],
            f"sanity check failed: {sanity}")

    info = environment(dev)
    info.update({
        "mnist_raw_sha256": mnist_hashes,
        "model_path": str(model_path),
        "model_file_sha256": sha256_file(model_path),
        "model_param_sha256": train_info["model_param_sha256"],
        "fp32_cpu_accuracy": train_info["fp32_cpu_accuracy"],
        "fp32_mps_accuracy": c32 / n,
        "fp16_mps_accuracy": c16 / n,
        "fp16_fp32_mps_diff_pp": diff_pp,
        "reference_logits_sha256": logits_sha,
        "reference_predictions_sha256": sha256_bytes(ref_preds.numpy().tobytes()),
        "double_eval_identical": True,
        "sanity_check": sanity,
    })
    print(f"fp32_mps {c32 / n:.4f}  fp16_mps {c16 / n:.4f}  diff {diff_pp} pp  sanity {sanity}")
    return m16, x16, y_test, ref_preds, info


def fmt_acc(correct):
    return f"{correct / 10000:.4f}"


# ---------------------------------------------------------------- step 4

def cmd_one_weight(args):
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    m16, x16, y, ref_preds, info = prepare(args.model)
    np.save(out / "reference_predictions.npy", ref_preds.numpy())

    w = m16.fc1.weight.detach().cpu().view(-1)
    g = rng("one_weight")
    draws = 0
    while True:
        idx = int(g.integers(0, w.numel()))
        draws += 1
        if abs(w[idx].item()) < 1:
            break

    with open(out / "one_weight.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["bit", "field", "original_value", "new_value", "class", "accuracy"])
        for bit in range(CFG["campaign"]["bits"]):
            old, new, cls, correct = inject(m16, "fc1.weight", idx, bit, x16, y, ref_preds)
            wr.writerow([bit, field(bit), repr(float(old)), repr(float(new)), cls, fmt_acc(correct)])

    n_in = m16.fc1.weight.shape[1]
    info.update({"element_index": idx, "hidden_unit": idx // n_in, "pixel": idx % n_in, "draws": draws})
    write_run_json(out, info, t0)
    print(f"one weight: fc1.weight[{idx}] (hidden unit {idx // n_in}, pixel {idx % n_in}) after {draws} draws")


# ---------------------------------------------------------------- step 5

def cmd_campaign(args):
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    m16, x16, y, ref_preds, info = prepare(args.model)
    np.save(out / "reference_predictions.npy", ref_preds.numpy())
    cc = CFG["campaign"]
    params = dict(m16.named_parameters())

    chosen = []  # (tensor, element_index, original_value), in draw order
    for name in cc["tensors"]:
        g = rng("sample_" + name.split(".")[0])
        idxs = g.choice(params[name].numel(), cc["elements_per_matrix"], replace=False)
        flat = params[name].detach().cpu().view(-1)
        chosen += [(name, int(i), flat[int(i)].item()) for i in idxs]
    with open(out / "chosen_elements.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["tensor", "element_index", "original_value"])
        wr.writerows([(name, i, repr(float(v))) for name, i, v in chosen])

    t_loop = time.perf_counter()
    totals = dict.fromkeys(CLASSES, 0)
    with open(out / "faults.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["tensor", "element_index", "bit", "original_value", "new_value", "class", "accuracy"])
        for name, i, _ in chosen:
            for bit in range(cc["bits"]):
                old, new, cls, correct = inject(m16, name, i, bit, x16, y, ref_preds)
                wr.writerow([name, i, bit, repr(float(old)), repr(float(new)), cls, fmt_acc(correct)])
                totals[cls] += 1

    info.update({"class_totals": totals, "campaign_loop_wall_time_s": round(time.perf_counter() - t_loop, 3)})
    write_run_json(out, info, t0)
    print(f"campaign done: {totals}")


# ---------------------------------------------------------------- step 6

def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summary_row(rows):
    n = len(rows)
    out = [n]
    for c in CLASSES:
        k = sum(r["class"] == c for r in rows)
        lo, hi = wilson(k, n)
        out += [k, f"{k / n:.6f}", f"{lo:.6f}", f"{hi:.6f}"]
    require(sum(out[1::4]) == n, "rows with an unknown class")
    return out


def cmd_summarize(args):
    run = Path(args.run_dir)
    outputs = [run / "per_bit.csv", run / "per_field.csv", run / "per_bit.png"]
    require(not any(p.exists() for p in outputs), f"summary files already exist in {run}")
    with open(run / "faults.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    header = ["n"] + [f"{c}_{s}" for c in CLASSES for s in ("count", "share", "lo", "hi")]
    bits = sorted({int(r["bit"]) for r in rows})

    per_bit = [[b, field(b)] + summary_row([r for r in rows if int(r["bit"]) == b]) for b in bits]
    with open(outputs[0], "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["bit", "field"] + header)
        wr.writerows(per_bit)
    with open(outputs[1], "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["field"] + header)
        for fld in ("sign", "exponent", "mantissa"):
            wr.writerow([fld] + summary_row([r for r in rows if field(int(r["bit"])) == fld]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {name: i for i, name in enumerate(["bit", "field"] + header)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(9.5, 14.5, color="#000000", alpha=0.04, lw=0)
    for start, end, label in ((-0.5, 9.5, "mantissa"), (9.5, 14.5, "exponent"), (14.5, 15.5, "sign")):
        ax.text((start + end) / 2, 1.04, label, ha="center", va="bottom", fontsize=9, color="#52514e")
    for x in (9.5, 14.5):
        ax.axvline(x, color="#52514e", lw=0.8, ls=":")
    for cls, color, label in (("sdc", "#2a78d6", "SDC"), ("non_finite", "#eb6834", "non-finite")):
        share = np.array([float(r[col[f"{cls}_share"]]) for r in per_bit])
        lo = np.array([float(r[col[f"{cls}_lo"]]) for r in per_bit])
        hi = np.array([float(r[col[f"{cls}_hi"]]) for r in per_bit])
        ax.errorbar(bits, share, yerr=[share - lo, hi - share], color=color, lw=2, marker="o", ms=5,
                    capsize=3, label=label)
    ax.set_xticks(bits)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("bit position (FP16)")
    ax.set_ylabel("share of faults (n = 300 per bit)")
    ax.set_title("Single-bit weight faults by bit position, 95% Wilson intervals", fontsize=10, pad=18)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(outputs[2], dpi=150)
    print(f"wrote {', '.join(p.name for p in outputs)} in {run}")


# ---------------------------------------------------------------- determinism checks

def cmd_compare(args):
    """Compare two runs of the same kind; writes logs/train_compare.json or logs/campaign_compare.json."""
    a, b = Path(args.run_a), Path(args.run_b)
    if (a / "faults.csv").exists():
        kind = "campaign"
        files = ["chosen_elements.csv", "faults.csv"]
        hashes = {f: [sha256_file(a / f), sha256_file(b / f)] for f in files}
    else:
        kind = "train"
        ra, rb = (json.loads((d / "run.json").read_text()) for d in (a, b))
        hashes = {k: [ra[k], rb[k]] for k in ("model_param_sha256", "model_file_sha256")}
    # Training is judged on the parameter hash (the file hash may depend on file metadata).
    compared = [hashes["model_param_sha256"]] if kind == "train" else hashes.values()
    identical = all(x == y for x, y in compared)
    report = {"run_a": str(a), "run_b": str(b), "sha256": hashes, "identical": identical}
    out = ROOT / "logs" / f"{kind}_compare.json"
    require(not out.exists(), f"{out} already exists")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if identical else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--run-name", required=True)
    for name in ("one-weight", "campaign"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--run-name", required=True)
    sub.add_parser("summarize").add_argument("run_dir")
    p = sub.add_parser("compare")
    p.add_argument("run_a")
    p.add_argument("run_b")
    args = ap.parse_args()
    {"train": cmd_train, "one-weight": cmd_one_weight, "campaign": cmd_campaign,
     "summarize": cmd_summarize, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
