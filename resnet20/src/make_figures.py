"""Figures for the Week 2 results write-up, from the official run logs.

Run from resnet20/:  .venv\\Scripts\\python.exe src/make_figures.py
Writes figures/fig1..fig5 (PNG) and figures/numbers.json (every number the write-up quotes from them).
Figure 5 re-evaluates the trained models (logs/*_train/model_fp32.pt, not in git) on the GPU and checks that
its predictions match the campaign's reference predictions before using them.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resnet20_fault import (ROOT, build_model, evaluate, load_data, normalize, require,  # noqa: E402
                            sha256_bytes, wilson)

LOGS = ROOT / "logs"
OUT = ROOT / "figures"
WEEK1 = ROOT.parent / "mnist" / "logs" / "campaign_a" / "per_bit.csv"
DATASETS = {"fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
DS_COLOR = {"fmnist": "#3b6fb6", "cifar10": "#d1495b"}
GROUPS = ["stage1", "stage2", "stage3", "fc"]
GROUP_LABEL = {"stage1": "stem + stage 1", "stage2": "stage 2", "stage3": "stage 3", "fc": "final layer (fc)"}
SMALL_PP = 0.1  # an SDC whose accuracy did not fall by more than this (10 of 10,000 images) counts as "small"
INK, GRID = "#52514e", "#e5e5e5"


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def field_bands(ax, top):
    for x in (9.5, 14.5):
        ax.axvline(x, color=INK, lw=0.8, ls=":")
    for start, end, label in ((-0.5, 9.5, "mantissa"), (9.5, 14.5, "exponent"), (14.5, 15.5, "sign")):
        ax.text((start + end) / 2, top, label, ha="center", va="bottom", fontsize=8, color=INK)


def faults(ds):
    with open(LOGS / f"{ds}_campaign_a" / "faults.csv", newline="") as f:
        return list(csv.DictReader(f))


def run_json(name):
    return json.loads((LOGS / name / "run.json").read_text())


def fig1_training(numbers):
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, label in DATASETS.items():
        hist = run_json(f"{ds}_train")["history"]
        ep = [h["epoch"] for h in hist]
        acc = [100 * h["test_accuracy"] for h in hist]
        ax.plot(ep, acc, color=DS_COLOR[ds], lw=1.6, label=label)
        numbers[f"{ds}_final_train_test_acc"] = acc[-1]
    for x in (80, 120):
        ax.axvline(x, color=INK, lw=0.8, ls=":")
        ax.text(x + 1, 41, "learning rate ÷10", fontsize=8, color=INK, rotation=90, va="bottom")
    ax.set_xlim(0, 161)
    ax.set_ylim(40, 100)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy, FP32 (%)")
    ax.set_title("Figure 1. Training: test accuracy after each epoch", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training.png", dpi=150)
    plt.close(fig)


def fig2_outcomes(numbers):
    cats = [("masked", "masked", "#d6d4ce"),
            ("sdc_small", f"SDC, accuracy fell ≤ {SMALL_PP} pp", "#9cc3ef"),
            ("sdc_large", f"SDC, accuracy fell > {SMALL_PP} pp", "#1f5fb0"),
            ("non_finite", "non-finite (inf or NaN)", "#eb6834")]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, (ds, label) in zip(axes, DATASETS.items()):
        ref = run_json(f"{ds}_campaign_a")["fp16_cuda_accuracy"]
        rows = faults(ds)
        counts = {c: np.zeros(16) for c, _, _ in cats}
        for r in rows:
            b, cls = int(r["bit"]), r["class"]
            if cls == "sdc":
                drop_pp = (ref - float(r["accuracy"])) * 100
                cls = "sdc_large" if drop_pp > SMALL_PP + 1e-9 else "sdc_small"
            counts[cls][b] += 1
        n = sum(counts[c] for c, _, _ in cats)
        require((n == n[0]).all(), "unequal faults per bit")
        bottom = np.zeros(16)
        for c, clabel, color in cats:
            share = 100 * counts[c] / n
            ax.bar(range(16), share, bottom=bottom, color=color, width=0.8, label=clabel)
            bottom += share
        numbers[f"{ds}_per_bit_counts"] = {c: counts[c].astype(int).tolist() for c, _, _ in cats}
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"share of faults (%)\n{int(n[0])} per bit")
        ax.set_title(f"{label}", fontsize=10, loc="left")
        field_bands(ax, 101)
        style(ax)
    axes[1].set_xticks(range(16))
    axes[1].set_xlabel("flipped bit position (FP16): 0 = lowest mantissa bit, 15 = sign")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.02))
    fig.suptitle("Figure 2. What each flipped bit did (each bar adds up to 100%)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(OUT / "fig2_outcomes_by_bit.png", dpi=150)
    plt.close(fig)


def per_bit_sdc(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([100 * float(r["sdc_share"]) for r in rows], [100 * float(r["sdc_lo"]) for r in rows],
            [100 * float(r["sdc_hi"]) for r in rows], int(rows[0]["n"]))


def fig3_sdc_vs_week1(numbers):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [(LOGS / f"{ds}_campaign_a" / "per_bit.csv", f"ResNet-20, {label}", DS_COLOR[ds], "o")
              for ds, label in DATASETS.items()]
    series.append((WEEK1, "Week 1 MLP, MNIST", "#7a7a7a", "s"))
    for k, (path, label, color, marker) in enumerate(series):
        share, lo, hi, n = per_bit_sdc(path)
        x = np.arange(16) + (k - 1) * 0.15
        ax.errorbar(x, share, yerr=[np.subtract(share, lo), np.subtract(hi, share)], color=color, lw=1.6,
                    marker=marker, ms=4, capsize=2, label=f"{label} (n = {n} per bit)")
        numbers[f"sdc_per_bit_{label}"] = share
    field_bands(ax, 101)
    ax.set_xticks(range(16))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("flipped bit position (FP16)")
    ax.set_ylabel("SDC share of faults (%)")
    ax.set_title("Figure 3. Silent data corruption by bit, with 95% Wilson intervals", fontsize=10, loc="left",
                 pad=16)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sdc_by_bit_vs_week1.png", dpi=150)
    plt.close(fig)


def fig4_groups(numbers):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    width = 0.38
    panels = (("sdc", "SDC (all bits)"),
              ("sdc_large", f"SDC with accuracy drop > {SMALL_PP} pp, or non-finite\n(note the 0–12% scale)"))
    for ax, (cls, title) in zip(axes, panels):
        for k, (ds, label) in enumerate(DATASETS.items()):
            ref = run_json(f"{ds}_campaign_a")["fp16_cuda_accuracy"]
            rows = faults(ds)
            shares, err = [], [[], []]
            for g in GROUPS:
                rg = [r for r in rows if r["group"] == g]
                if cls == "sdc":
                    hit = sum(r["class"] == "sdc" for r in rg)
                else:
                    hit = sum(r["class"] == "non_finite"
                              or (r["class"] == "sdc" and (ref - float(r["accuracy"])) * 100 > SMALL_PP + 1e-9)
                              for r in rg)
                lo, hi = wilson(hit, len(rg))
                p = hit / len(rg)
                shares.append(100 * p)
                err[0].append(100 * (p - lo))
                err[1].append(100 * (hi - p))
                numbers.setdefault(f"group_{cls}", {})[f"{ds}_{g}"] = [hit, len(rg)]
            x = np.arange(len(GROUPS)) + (k - 0.5) * width
            ax.bar(x, shares, width=width, color=DS_COLOR[ds], label=label, yerr=err, capsize=3,
                   error_kw={"elinewidth": 1, "ecolor": INK})
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS], fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        style(ax)
    axes[0].set_ylabel("share of faults (%), 2,400 per group")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 12)  # note: different scale from the left panel
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 4. Faults by where the weight sits in the network (input side → output side)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_by_group.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def top2_gaps(ds):
    """Gap between the largest and second-largest FP16 output score of each clean test image."""
    train = run_json(f"{ds}_train")
    camp = run_json(f"{ds}_campaign_a")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _, _, x_test, _, raw = load_data(train["dataset"])
    require(raw == train["raw_sha256"], "dataset files differ from the training run")
    model = build_model(x_test.shape[1])
    model.load_state_dict(torch.load(LOGS / f"{ds}_train" / "model_fp32.pt", map_location="cpu", weights_only=True))
    model.eval()
    m16 = model.half().to("cuda")
    x16 = normalize(x_test.to("cuda"), train["normalize_mean"], train["normalize_std"]).half()
    logits = evaluate(m16, x16)
    require(sha256_bytes(logits.numpy().tobytes()) == camp["reference_logits_sha256"],
            f"{ds}: clean logits differ from the campaign's reference")
    top2 = logits.float().topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).numpy()


def fig5_gaps(numbers):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ds, label in DATASETS.items():
        gap = top2_gaps(ds)
        numbers[f"{ds}_gaps"] = {"exact_ties": int((gap == 0).sum()), "lt_0.01": int((gap < 0.01).sum()),
                                 "lt_0.1": int((gap < 0.1).sum()), "lt_1": int((gap < 1).sum()),
                                 "median": float(np.median(gap)), "n": int(gap.size)}
        g = np.sort(np.maximum(gap, 1e-3))  # an exact tie (gap 0) is drawn at the left edge
        ax.step(g, np.arange(1, g.size + 1), where="post", color=DS_COLOR[ds], lw=1.8, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 100)
    ax.set_ylim(0.8, 12000)
    for x in (0.01, 0.1, 1):
        ax.axvline(x, color=INK, lw=0.6, ls=":")
    ax.set_xlabel("gap between the top two output scores (log scale)")
    ax.set_ylabel("number of test images (of 10,000)\nwith a gap this small or smaller (log scale)")
    ax.set_title("Figure 5. How close each clean test image is to changing its answer (FP16, no fault)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_top2_gap.png", dpi=150)
    plt.close(fig)


def main():
    require(torch.cuda.is_available(), "Figure 5 needs CUDA")
    OUT.mkdir(exist_ok=True)
    numbers = {}
    for make in (fig1_training, fig2_outcomes, fig3_sdc_vs_week1, fig4_groups, fig5_gaps):
        make(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(f"wrote 5 figures and numbers.json in {OUT}")


if __name__ == "__main__":
    main()
