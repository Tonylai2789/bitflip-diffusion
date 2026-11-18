"""ResNet-20 bit-flip study (plan-resnet20-week2.md, section 6). Run from resnet20/ with .venv\\Scripts\\python.exe:

    python src/resnet20_fault.py train --dataset fashion_mnist --run-name fmnist_train
    python src/resnet20_fault.py one-weight --model logs/fmnist_train/model_fp32.pt --run-name fmnist_one_weight
    python src/resnet20_fault.py campaign --model logs/fmnist_train/model_fp32.pt --run-name fmnist_campaign_a
    python src/resnet20_fault.py summarize logs/fmnist_campaign_a
    python src/resnet20_fault.py compare logs/fmnist_campaign_a logs/fmnist_campaign_b

Datasets: fashion_mnist, cifar10. Settings come from configs/resnet20.json. Each run writes logs/<run_name>/ and
refuses to overwrite it. GPU training is not bit-repeatable (Week2 tab), so each dataset is trained once and every
later run loads the saved weights; the dataset and normalization come from the training run's run.json.
Structure follows ../mnist/src/mnist_fault.py.
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
from datetime import datetime
from fnmatch import fnmatch
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitflip import field, fingerprint, flip, restore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # resnet20/
CFG = json.loads((ROOT / "configs" / "resnet20.json").read_text())
DEVICE = "cuda"
CLASSES = ["masked", "sdc", "non_finite"]
MNIST_RAW = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
DATASETS = {  # name: (torchvision class, raw files hashed into run.json, relative to data/)
    "fashion_mnist": (torchvision.datasets.FashionMNIST, ["FashionMNIST/raw/" + f for f in MNIST_RAW]),
    "cifar10": (torchvision.datasets.CIFAR10, ["cifar-10-python.tar.gz"]),
}
# (faultable weights, all parameters) by input channels; plan section 2.
EXPECTED_COUNTS = {1: (268_048, 269_434), 3: (268_336, 269_722)}


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


def environment():
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    return {
        "command": shlex.join(sys.argv),
        "git_commit": commit,
        # Same convention as ../cifar10 and ../mnist: untracked files (earlier logs/) don't count.
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain", "--untracked-files=no"])) if commit else None,
        "master_seed": CFG["master_seed"],
        "roles": CFG["roles"],
        "derived_seeds": derived_seeds(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_driver": run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ["torch", "torchvision", "numpy"]},
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


# ---------------------------------------------------------------- data

def load_data(name):
    """Train and test images as float32 (N, C, H, W) in [0, 1] plus int64 labels, on CPU; SHA-256 of the raw files."""
    require(name in DATASETS, f"unknown dataset {name}")
    cls, raw_files = DATASETS[name]
    root = ROOT / CFG["data"]["root"]
    out = []
    for train in (True, False):
        ds = cls(root=root, train=train, download=True)
        x = torch.as_tensor(ds.data)  # Fashion-MNIST (N, 28, 28) uint8; CIFAR-10 (N, 32, 32, 3) uint8
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        out += [x.float().div(255).contiguous(), torch.as_tensor(ds.targets, dtype=torch.long)]
    return (*out, {f: sha256_file(root / f) for f in raw_files})


def norm_stats(x_train):
    """Per-channel mean and std over the whole training set, in float64 (D6)."""
    chans = [x_train[:, c].double() for c in range(x_train.shape[1])]
    return [float(c.mean()) for c in chans], [float(c.std()) for c in chans]


def normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def augment(x, crop_pad, hflip):
    """Per-image random horizontal flip and random crop after zero padding, on x's device.
    Randomness comes from the global CPU torch RNG (seeded by the training role)."""
    n, c, h, w = x.shape
    if hflip:
        m = (torch.rand(n) < 0.5).to(x.device)
        x = torch.where(m.view(n, 1, 1, 1), x.flip(3), x)
    if crop_pad:
        p = crop_pad
        xp = F.pad(x, (p, p, p, p))
        rows = torch.randint(0, 2 * p + 1, (n,)).to(x.device).view(n, 1) + torch.arange(h, device=x.device)
        cols = torch.randint(0, 2 * p + 1, (n,)).to(x.device).view(n, 1) + torch.arange(w, device=x.device)
        x = xp[torch.arange(n, device=x.device).view(n, 1, 1, 1), torch.arange(c, device=x.device).view(1, c, 1, 1),
               rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]
    return x


# ---------------------------------------------------------------- model

class Block(nn.Module):
    """Basic block: conv-BN-ReLU-conv-BN plus an option-A shortcut, then ReLU. Option A (He et al. 2016, CIFAR):
    identity, or for a stride-2 block every second pixel with the new channels zero-padded; it has no weights."""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.stride, self.extra = stride, c_out - c_in

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        short = x
        if self.stride != 1 or self.extra:
            short = F.pad(x[:, :, ::self.stride, ::self.stride],
                          (0, 0, 0, 0, self.extra // 2, self.extra - self.extra // 2))
        return F.relu(out + short)


class ResNet20(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        m = CFG["model"]
        widths = m["widths"]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        stages, c = [], widths[0]
        for i, width in enumerate(widths):
            blocks = []
            for j in range(m["blocks_per_stage"]):
                blocks.append(Block(c, width, 2 if i > 0 and j == 0 else 1))
                c = width
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(widths[-1], m["num_classes"])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(x.mean(dim=(2, 3)))  # global average pool


def build_model(in_channels):
    m = CFG["model"]
    require((m["name"], m["blocks_per_stage"], m["widths"], m["shortcut"]) == ("resnet20", 3, [16, 32, 64], "A"),
            f"unsupported model settings {m}")
    model = ResNet20(in_channels)
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight)  # He initialization; BN and the fc bias keep PyTorch defaults
    n_weights = sum(p.numel() for names in groups(model).values() for p in map(model.get_parameter, names))
    n_params = sum(p.numel() for p in model.parameters())
    require((n_weights, n_params) == EXPECTED_COUNTS[in_channels],
            f"weights {n_weights}, parameters {n_params}, expected {EXPECTED_COUNTS[in_channels]}")
    return model


def groups(model):
    """{group: [parameter names]} from the config patterns. Every conv and linear weight is in exactly one group;
    biases and batch norm are in none (D2, D3)."""
    names = [n for n, _ in model.named_parameters()]
    out = {g: [n for n in names if any(fnmatch(n, p) for p in pats)] for g, pats in CFG["campaign"]["groups"].items()}
    weights = [n + ".weight" for n, mod in model.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    grouped = [n for ns in out.values() for n in ns]
    require(sorted(grouped) == sorted(weights), f"groups {out} do not cover exactly the conv and linear weights")
    return out


def locate(model, names, k):
    """Flat index k across the concatenated tensors `names` (in order) -> (tensor name, element index)."""
    for name in names:
        n = model.get_parameter(name).numel()
        if k < n:
            return name, k
        k -= n
    raise IndexError(k)


def param_hash(model):
    h = hashlib.sha256()
    for t in model.state_dict().values():
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, x):
    """Logits for all test images in one batch, copied to CPU (this also synchronizes the GPU)."""
    return model(x).cpu()


def correct_count(logits, y):
    return int((logits.argmax(dim=1) == y.cpu()).sum())


# ---------------------------------------------------------------- train

def cmd_train(args):
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    tc = CFG["train"]
    require((tc["device"], tc["dtype"], tc["optimizer"], tc["loss"]) == (DEVICE, "float32", "sgd", "cross_entropy"),
            f"unsupported train settings {tc}")
    aug = tc["augment"][args.dataset]
    require(set(aug) <= {"random_crop_pad4", "horizontal_flip"}, f"unknown augmentation {aug}")
    torch.backends.cudnn.benchmark = True  # training is not bit-repeatable anyway (Week2 tab); recorded below

    x_train, y_train, x_test, y_test, raw_hashes = load_data(args.dataset)
    mean, std = norm_stats(x_train)
    x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
    x_test, y_test = normalize(x_test.to(DEVICE), mean, std), y_test.to(DEVICE)
    seed = derived_seeds()["train_" + args.dataset]

    torch.manual_seed(seed)  # initial weights, batch order and augmentation all draw from this
    model = build_model(x_train.shape[1]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=tc["lr_milestones"], gamma=tc["lr_gamma"])
    n, bs = x_train.shape[0], tc["batch_size"]
    history = []
    t_train = time.perf_counter()
    for epoch in range(tc["epochs"]):
        model.train()
        perm = torch.randperm(n).to(DEVICE)
        total = torch.zeros((), device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = normalize(augment(x_train[idx], 4 if "random_crop_pad4" in aug else 0, "horizontal_flip" in aug),
                           mean, std)
            loss = F.cross_entropy(model(xb), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        sched.step()
        model.eval()
        acc = correct_count(evaluate(model, x_test), y_test) / len(y_test)
        history.append({"epoch": epoch + 1, "mean_loss": round(float(total) / n, 6), "test_accuracy": acc})
        print(f"epoch {epoch + 1}: mean loss {float(total) / n:.4f}  test acc {acc:.4f}  "
              f"({time.perf_counter() - t_train:.0f}s)", flush=True)
    train_wall = time.perf_counter() - t_train

    model.eval()
    c32 = correct_count(evaluate(model, x_test), y_test)
    c16 = correct_count(evaluate(copy.deepcopy(model).half(), x_test.half()), y_test)
    torch.save(model.state_dict(), out / "model_fp32.pt")
    info = environment()
    info.update({
        "dataset": args.dataset,
        "loader": f"torchvision.datasets.{DATASETS[args.dataset][0].__name__}, whole set as tensors on the GPU "
                  "(no DataLoader); pixels / 255, then per-channel normalization",
        "raw_sha256": raw_hashes,
        "normalize_mean": mean,
        "normalize_std": std,
        "train_settings": tc,
        "augment": aug,
        "model_file_sha256": sha256_file(out / "model_fp32.pt"),
        "model_param_sha256": param_hash(model),
        "fp32_cuda_accuracy": c32 / len(y_test),
        "fp16_cuda_accuracy": c16 / len(y_test),
        "train_wall_time_s": round(train_wall, 3),
        "history": history,
    })
    write_run_json(out, info, t0)
    print(f"fp32 {c32 / len(y_test):.4f}  fp16 {c16 / len(y_test):.4f}  param hash {info['model_param_sha256']}")


# ---------------------------------------------------------------- shared by one-weight and campaign

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
    p = m16.get_parameter(name)
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
    """FP32 and FP16 copies on CUDA, FP16 margin check, double-eval check, fault-free reference, sanity check."""
    ec = CFG["eval"]
    require((ec["device"], ec["dtype"], ec["cudnn_deterministic"]) == (DEVICE, "float16", True),
            f"unsupported eval settings {ec}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_info = json.loads((Path(model_path).parent / "run.json").read_text())
    _, _, x_test, y_test, raw_hashes = load_data(train_info["dataset"])
    require(raw_hashes == train_info["raw_sha256"], "dataset files differ from the training run")
    model = build_model(x_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    require(param_hash(model) == train_info["model_param_sha256"], "model does not match its training run.json")
    model.eval()
    m32 = copy.deepcopy(model).to(DEVICE)
    m16 = copy.deepcopy(model).half().to(DEVICE)
    x32 = normalize(x_test.to(DEVICE), train_info["normalize_mean"], train_info["normalize_std"])
    x16 = x32.half()
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

    # Sanity check (no RNG): bit 14 of the fc.weight element with the largest FP16 magnitude below 1.
    w = m16.fc.weight.detach().cpu().view(-1).abs()
    idx = int(torch.where(w < 1, w, torch.full_like(w, -1)).argmax())
    fp = fingerprint(m16.fc.weight)
    old = m16.fc.weight.data.view(-1)[idx].item()
    pattern = flip(m16, "fc.weight", idx, 14)
    try:
        new = m16.fc.weight.data.view(-1)[idx].item()
        flipped = evaluate(m16, x16)
    finally:
        restore(m16, "fc.weight", idx, pattern)
    restored = evaluate(m16, x16)
    sanity = {
        "tensor": "fc.weight", "element_index": idx, "bit": 14,
        "original_value": repr(old), "new_value": repr(new),
        "logits_changed": not torch.equal(flipped, ref_logits),
        "predictions_changed": int((flipped.argmax(dim=1) != ref_preds).sum()),
        "logits_finite": bool(torch.isfinite(flipped).all()),
        "accuracy_flipped": correct_count(flipped, y_test) / n,
        "restored_logits_match": sha256_bytes(restored.numpy().tobytes()) == logits_sha,
        "restored_fingerprint_match": fingerprint(m16.fc.weight) == fp,
    }
    require(sanity["logits_changed"] and sanity["restored_logits_match"] and sanity["restored_fingerprint_match"],
            f"sanity check failed: {sanity}")

    info = environment()
    info.update({
        "dataset": train_info["dataset"],
        "raw_sha256": raw_hashes,
        "normalize_mean": train_info["normalize_mean"],
        "normalize_std": train_info["normalize_std"],
        "model_path": str(model_path),
        "model_file_sha256": sha256_file(model_path),
        "model_param_sha256": train_info["model_param_sha256"],
        "fp32_cuda_accuracy": c32 / n,
        "fp16_cuda_accuracy": c16 / n,
        "fp16_fp32_diff_pp": diff_pp,
        "reference_logits_sha256": logits_sha,
        "reference_predictions_sha256": sha256_bytes(ref_preds.numpy().tobytes()),
        "double_eval_identical": True,
        "sanity_check": sanity,
    })
    print(f"{train_info['dataset']}: fp32 {c32 / n:.4f}  fp16 {c16 / n:.4f}  diff {diff_pp} pp  sanity {sanity}")
    return m16, x16, y_test, ref_preds, info


def fmt_acc(correct, n):
    return f"{correct / n:.4f}"


# ---------------------------------------------------------------- one weight

def cmd_one_weight(args):
    """One element, uniform over all faultable weights (all groups), redrawn until |w| < 1, through all 16 bits."""
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    m16, x16, y, ref_preds, info = prepare(args.model)
    np.save(out / "reference_predictions.npy", ref_preds.numpy())

    names = [n for ns in groups(m16).values() for n in ns]
    total = sum(m16.get_parameter(n).numel() for n in names)
    g = rng("one_weight")
    draws = 0
    while True:
        name, idx = locate(m16, names, int(g.integers(0, total)))
        draws += 1
        if abs(m16.get_parameter(name).data.view(-1)[idx].item()) < 1:
            break

    with open(out / "one_weight.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["bit", "field", "original_value", "new_value", "class", "accuracy"])
        for bit in range(CFG["campaign"]["bits"]):
            old, new, cls, correct = inject(m16, name, idx, bit, x16, y, ref_preds)
            wr.writerow([bit, field(bit), repr(float(old)), repr(float(new)), cls, fmt_acc(correct, len(y))])

    info.update({"tensor": name, "element_index": idx, "draws": draws})
    write_run_json(out, info, t0)
    print(f"one weight: {name}[{idx}] after {draws} draws")


# ---------------------------------------------------------------- campaign

def cmd_campaign(args):
    t0 = time.perf_counter()
    out = new_run_dir(args.run_name)
    m16, x16, y, ref_preds, info = prepare(args.model)
    np.save(out / "reference_predictions.npy", ref_preds.numpy())
    cc = CFG["campaign"]
    grp = groups(m16)

    chosen = []  # (group, tensor, element_index, original_value), in draw order
    for gname, names in grp.items():
        total = sum(m16.get_parameter(n).numel() for n in names)
        for k in rng("sample_" + gname).choice(total, cc["elements_per_group"], replace=False):
            name, i = locate(m16, names, int(k))
            chosen.append((gname, name, i, m16.get_parameter(name).data.view(-1)[i].item()))
    with open(out / "chosen_elements.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["group", "tensor", "element_index", "original_value"])
        wr.writerows([(gname, name, i, repr(float(v))) for gname, name, i, v in chosen])

    t_loop = time.perf_counter()
    totals = dict.fromkeys(CLASSES, 0)
    with open(out / "faults.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["group", "tensor", "element_index", "bit", "original_value", "new_value", "class", "accuracy"])
        for k, (gname, name, i, _) in enumerate(chosen):
            for bit in range(cc["bits"]):
                old, new, cls, correct = inject(m16, name, i, bit, x16, y, ref_preds)
                wr.writerow([gname, name, i, bit, repr(float(old)), repr(float(new)), cls, fmt_acc(correct, len(y))])
                totals[cls] += 1
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(chosen)} elements  {totals}  ({time.perf_counter() - t_loop:.0f}s)", flush=True)

    info.update({"groups": grp, "class_totals": totals,
                 "campaign_loop_wall_time_s": round(time.perf_counter() - t_loop, 3)})
    write_run_json(out, info, t0)
    print(f"campaign done: {totals}")


# ---------------------------------------------------------------- summarize

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
    outputs = [run / "per_bit.csv", run / "per_field.csv", run / "per_group.csv", run / "per_bit.png"]
    require(not any(p.exists() for p in outputs), f"summary files already exist in {run}")
    dataset = json.loads((run / "run.json").read_text())["dataset"]
    with open(run / "faults.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    header = ["n"] + [f"{c}_{s}" for c in CLASSES for s in ("count", "share", "lo", "hi")]
    bits = sorted({int(r["bit"]) for r in rows})

    per_bit = [[b, field(b)] + summary_row([r for r in rows if int(r["bit"]) == b]) for b in bits]
    tables = [
        (["bit", "field"], per_bit),
        (["field"], [[fld] + summary_row([r for r in rows if field(int(r["bit"])) == fld])
                     for fld in ("sign", "exponent", "mantissa")]),
        (["group"], [[g] + summary_row([r for r in rows if r["group"] == g]) for g in CFG["campaign"]["groups"]]),
    ]
    for path, (keys, table) in zip(outputs, tables):
        with open(path, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(keys + header)
            wr.writerows(table)

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
    ax.set_ylabel(f"share of faults (n = {per_bit[0][2]} per bit)")
    ax.set_title(f"ResNet-20 on {dataset}: single-bit weight faults by bit position, 95% Wilson intervals",
                 fontsize=10, pad=18)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(outputs[3], dpi=150)
    print(f"wrote {', '.join(p.name for p in outputs)} in {run}")


# ---------------------------------------------------------------- determinism check

def cmd_compare(args):
    """Compare two campaigns; writes logs/compare_<a>_vs_<b>.json. (Training is run once, so it is not compared.)"""
    a, b = Path(args.run_a), Path(args.run_b)
    require((a / "faults.csv").exists() and (b / "faults.csv").exists(), "compare takes two campaign run folders")
    files = ["chosen_elements.csv", "faults.csv"]
    hashes = {f: [sha256_file(a / f), sha256_file(b / f)] for f in files}
    identical = all(x == y for x, y in hashes.values())
    report = {"run_a": str(a), "run_b": str(b), "sha256": hashes, "identical": identical}
    out = ROOT / "logs" / f"compare_{a.name}_vs_{b.name}.json"
    require(not out.exists(), f"{out} already exists")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if identical else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
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
    require(torch.cuda.is_available(), "CUDA is not available; this study runs on the RTX 5090 only")
    {"train": cmd_train, "one-weight": cmd_one_weight, "campaign": cmd_campaign,
     "summarize": cmd_summarize, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
