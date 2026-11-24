"""Where the damage from single-bit FP16 weight faults comes from (analysis-damage-2026-11-23.md).

Run from resnet20/ with .venv\\Scripts\\python.exe:

    python src/analyze_damage.py logs/fmnist_campaign_a/faults.csv logs/cifar10_campaign_a/faults.csv
    python src/analyze_damage.py logs/fmnist_campaign_a/faults.csv logs/cifar10_campaign_a/faults.csv --figures figures

Takes any number of faults.csv files with the campaign's columns (the sampled campaigns or an exhaustive run).
Each file needs the run.json written next to it: the fault-free FP16 accuracy, the dataset and the model path come
from there. Prints markdown tables to stdout. Memory stays at roughly 25 bytes per fault row.

Definitions as in make_figures.py: "large SDC" = SDC whose accuracy fell by more than 0.1 pp; "damaging" = large SDC
or non-finite. Intervals are 95% cluster-bootstrap intervals over weights (the 16 bits of one weight are not
independent faults); a unit whose rows cover every one of its weights is a census and is marked "exact".
The fc check (section Q3e) needs CUDA; without it, that section is skipped with a note.
"""
import argparse
import array
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resnet20_fault import (ROOT, build_model, evaluate, groups, load_data, normalize, require,  # noqa: E402
                            sha256_bytes)

COLUMNS = ["group", "tensor", "element_index", "bit", "original_value", "new_value", "class", "accuracy"]
CLASS_CODE = {"masked": 0, "sdc": 1, "non_finite": 2}
SMALL_PP = 0.1
FP16_INF_AT = 65520.0  # FP16 max is 65504; anything that rounds at or above 65520 becomes inf
N_BOOT = 2000
BOOT_CELLS = 4_000_000  # cap on resamples x weights held in memory at once
DATASET_LABEL = {"fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}


# ---------------------------------------------------------------- loading

def load_faults(path):
    """Stream faults.csv into compact numpy columns. Group and tensor names become small integer codes."""
    gcode, tcode = {}, {}
    g, t, e, b = array.array("b"), array.array("h"), array.array("i"), array.array("b")
    w, wn, c, acc = array.array("f"), array.array("f"), array.array("b"), array.array("f")
    with open(path, newline="") as f:
        rd = csv.reader(f)
        require(next(rd) == COLUMNS, f"{path}: unexpected columns")
        for row in rd:
            g.append(gcode.setdefault(row[0], len(gcode)))
            t.append(tcode.setdefault(row[1], len(tcode)))
            e.append(int(row[2]))
            b.append(int(row[3]))
            w.append(float(row[4]))
            wn.append(float(row[5]))
            c.append(CLASS_CODE[row[6]])
            acc.append(float(row[7]))
    cols = {k: np.frombuffer(v, dtype=v.typecode).copy()
            for k, v in (("g", g), ("t", t), ("e", e), ("bit", b), ("w", w), ("w_new", wn), ("cls", c), ("acc", acc))}
    return cols, list(gcode), list(tcode)


def load_context(csv_path):
    run = csv_path.parent / "run.json"
    require(run.exists(), f"{run} not found: the script needs the run.json written next to faults.csv")
    info = json.loads(run.read_text())
    mp = Path(info["model_path"])
    mp = mp if mp.is_absolute() else ROOT / mp
    sd = torch.load(mp, map_location="cpu", weights_only=True)
    model = build_model(sd["conv1.weight"].shape[1])
    model.load_state_dict(sd)
    model.eval()
    grp = groups(model)
    return info, model, sd, grp


# ---------------------------------------------------------------- statistics helpers

def per_weight(keys, *flags):
    """Per-weight row count and flag counts. keys: one int64 id per row (tensor code << 32 | element index)."""
    _, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    return n, [np.bincount(inv, weights=f.astype(np.float64)) for f in flags]


def boot(n_w, k_w, rng):
    """Bootstrap replicates of sum(k)/sum(n), resampling weights with replacement."""
    m = len(n_w)
    out = np.empty(N_BOOT)
    step = max(1, BOOT_CELLS // m)
    for s in range(0, N_BOOT, step):
        idx = rng.integers(0, m, size=(min(step, N_BOOT - s), m))
        out[s:s + idx.shape[0]] = k_w[idx].sum(1) / n_w[idx].sum(1)
    return out


class Unit:
    """Rows belonging to one tensor or group, with per-weight counts for the cluster bootstrap."""

    def __init__(self, d, rows, size, rng):
        self.rows = rows
        self.size = size
        keys = (d["t"][rows].astype(np.int64) << 32) | d["e"][rows].astype(np.int64)
        flags = (d["sdc"][rows], d["large"][rows], d["dmg"][rows], d["cls"][rows] == 2,
                 d["dmg"][rows] & (d["bit"][rows] == 14))
        self.n_w, ks = per_weight(keys, *flags)
        self.k = dict(zip(("sdc", "large", "dmg", "nonfin", "dmg14"), ks))
        self.weights = len(self.n_w)
        self.faults = int(self.n_w.sum())
        self.exact = self.weights == size and (self.n_w == 16).all()
        self.rng = rng
        self._reps = {}

    def count(self, what):
        return int(round(self.k[what].sum()))

    def share(self, what):
        return self.k[what].sum() / self.n_w.sum()

    def reps(self, what):
        if what not in self._reps:
            self._reps[what] = boot(self.n_w, self.k[what], self.rng)
        return self._reps[what]

    def ci(self, what):
        if self.exact or self.weights < 2:
            return None
        return tuple(np.percentile(self.reps(what), [2.5, 97.5]))

    def fmt(self, what, scale=100, digits=1):
        p = self.share(what) * scale
        if self.exact:
            return f"{p:.{digits}f} (exact)"
        lo_hi = self.ci(what)
        if lo_hi is None:
            return f"{p:.{digits}f} [n/a]"
        return f"{p:.{digits}f} [{lo_hi[0] * scale:.{digits}f}, {lo_hi[1] * scale:.{digits}f}]"


def avg_rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x, y):
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0, 1])


def pct(k, n, digits=1):
    return f"{100 * k / n:.{digits}f}" if n else "–"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(v) for v in r) + " |")
    print()


BINS = [(-np.inf, SMALL_PP + 1e-9, "≤ 0.1 pp or rose"), (SMALL_PP + 1e-9, 1, "0.1–1"), (1, 5, "1–5"),
        (5, 20, "5–20"), (20, 50, "20–50"), (50, np.inf, "> 50")]
BIN_HEADER = [f"drop {name}" for _, _, name in BINS] + ["non-finite"]


def bin_counts(d, r):
    """Finite faults among rows r counted by accuracy-drop bin, then the non-finite count."""
    fin = r[d["cls"][r] != 2]
    return [int(((d["drop"][fin] > lo) & (d["drop"][fin] <= hi)).sum()) for lo, hi, _ in BINS] + \
        [int((d["cls"][r] == 2).sum())]


def drop_summary(x):
    return f"{np.median(x):.2f}" if len(x) else "–"


# ---------------------------------------------------------------- analysis of one file

def analyze(csv_path, fig_dir):
    info, model, sd, grp = load_context(csv_path)
    d, gnames, tnames = load_faults(csv_path)
    ref = info["fp16_cuda_accuracy"]
    label = DATASET_LABEL.get(info["dataset"], info["dataset"])
    rng = np.random.default_rng(20261123)

    d["drop"] = (ref - d["acc"].astype(np.float64)) * 100  # pp; negative = accuracy rose
    d["sdc"] = d["cls"] == 1
    d["large"] = d["sdc"] & (d["drop"] > SMALL_PP + 1e-9)
    d["dmg"] = d["large"] | (d["cls"] == 2)
    n_all = len(d["cls"])

    group_order = list(grp)
    tensor_order = [t for g in group_order for t in grp[g]]
    tsize = {t: model.get_parameter(t).numel() for t in tensor_order}
    gsize = {g: sum(tsize[t] for t in grp[g]) for g in group_order}
    tensor_group = {t: g for g in group_order for t in grp[g]}
    require(set(gnames) <= set(group_order) and set(tnames) <= set(tensor_order), "unknown group or tensor names")
    gidx = {name: i for i, name in enumerate(gnames)}
    tidx = {name: i for i, name in enumerate(tnames)}
    g_units = {g: Unit(d, np.flatnonzero(d["g"] == gidx[g]), gsize[g], rng) for g in group_order if g in gidx}
    t_units = {t: Unit(d, np.flatnonzero(d["t"] == tidx[t]), tsize[t], rng) for t in tensor_order if t in tidx}

    print(f"# {label}: {csv_path}\n")
    print(f"{n_all:,} faults on {sum(u.weights for u in g_units.values()):,} weights. Fault-free FP16 accuracy "
          f"{ref:.4f}. Masked {int((d['cls'] == 0).sum()):,}, SDC {int(d['sdc'].sum()):,} "
          f"(large {int(d['large'].sum()):,}), non-finite {int((d['cls'] == 2).sum()):,}, "
          f"damaging {int(d['dmg'].sum()):,} ({pct(d['dmg'].sum(), n_all, 2)}%).\n")

    # ------------------------------------------------ Q1: size against faults
    print("## Q1. Layer size against faults\n")
    print("Per group (interval: 95% bootstrap over weights).\n")
    table(["group", "weights in group", "weights sampled", "faults", "SDC %", "damaging %", "non-finite",
           "bit-14 damaging"],
          [[g, f"{gsize[g]:,}", u.weights, u.faults, u.fmt("sdc"), u.fmt("dmg"), u.count("nonfin"),
            f"{u.count('dmg14')} of {int((d['bit'][u.rows] == 14).sum())}"] for g, u in g_units.items()])
    print("Per tensor. 'expected' = weights a uniform draw of 150 per group puts in this tensor on average.\n")
    rows = []
    for t, u in t_units.items():
        g = tensor_group[t]
        exp = g_units[g].weights * tsize[t] / gsize[g]
        n14 = int((d["bit"][u.rows] == 14).sum())
        rows.append([t, f"{tsize[t]:,}", f"{u.weights} ({exp:.1f})", u.fmt("sdc"), u.fmt("dmg"),
                     u.count("dmg"), f"{u.count('dmg14')} of {n14}"])
    table(["tensor", "size", "weights sampled (expected)", "SDC %", "damaging %", "damaging faults",
           "bit-14 damaging"], rows)
    ts = list(t_units)
    sizes = [tsize[t] for t in ts]
    print(f"Spearman rank correlation over {len(ts)} tensors: size vs SDC share "
          f"{spearman(sizes, [t_units[t].share('sdc') for t in ts]):+.2f}; size vs damaging share "
          f"{spearman(sizes, [t_units[t].share('dmg') for t in ts]):+.2f}. Convolutions only "
          f"({len(ts) - 1} tensors, without fc): SDC {spearman(sizes[:-1], [t_units[t].share('sdc') for t in ts[:-1]]):+.2f},"
          f" damaging {spearman(sizes[:-1], [t_units[t].share('dmg') for t in ts[:-1]]):+.2f}.\n")
    print("Size with depth held fixed: in stages 2 and 3, `layerN.0.conv1` has half the weights of the other convs.\n")
    rows = []
    for g in ("stage2", "stage3"):
        if g not in g_units:
            continue
        first = [t for t in grp[g] if t.endswith(".0.conv1.weight")]
        rest = [t for t in grp[g] if t not in first]
        for name, ts_ in (("first conv (half size)", first), ("other convs", rest)):
            rr = np.concatenate([t_units[t].rows for t in ts_ if t in t_units])
            u = Unit(d, rr, sum(tsize[t] for t in ts_), rng)
            rows.append([g, name, f"{tsize[ts_[0]]:,} each", u.weights, u.fmt("sdc"), u.fmt("dmg")])
    table(["group", "tensors", "size", "weights sampled", "SDC %", "damaging %"], rows)

    # ------------------------------------------------ Q2: how many faults do damage
    print("## Q2. How many faults do damage\n")
    rows = []
    for bit in range(16):
        m = d["bit"] == bit
        rows.append([bit, int(m.sum()), int((m & d["large"]).sum()), int((m & (d["cls"] == 2)).sum()),
                     int((m & d["dmg"]).sum()), pct((m & d["dmg"]).sum(), m.sum()),
                     pct((m & d["dmg"]).sum(), d["dmg"].sum())])
    table(["bit", "faults", "large SDC", "non-finite", "damaging", "% of this bit's faults",
           "% of all damaging"], rows)
    print("Damaging faults by group and bit field.\n")
    rows = []
    for g, u in g_units.items():
        r, b = u.rows, d["bit"][u.rows]
        dm = d["dmg"][r]
        rows.append([g, int(dm.sum()), int((dm & (b == 14)).sum()), int((dm & (b == 15)).sum()),
                     int((dm & (b >= 10) & (b <= 13)).sum()), int((dm & (b <= 9)).sum()),
                     pct(dm.sum(), d["dmg"].sum())])
    table(["group", "damaging", "bit 14", "bit 15 (sign)", "bits 10–13", "bits 0–9", "% of all damaging"], rows)
    other = d["dmg"] & (d["bit"] != 14)
    if other.any():
        print("Damaging faults outside bit 14, by bit and group:\n")
        bits = sorted(set(d["bit"][other].tolist()))
        table(["group"] + [f"bit {b}" for b in bits],
              [[g] + [int((other[u.rows] & (d["bit"][u.rows] == b)).sum()) for b in bits]
               for g, u in g_units.items()])
    total = sum(gsize[g] for g in g_units)
    print(f"Groups were sampled equally, so the overall {pct(d['dmg'].sum(), n_all, 2)}% treats fc (640 weights) "
          "like stage 3. Weighting each group by its real size gives the chance that a fault at a uniformly random "
          "weight and bit of the whole network does damage, and where those damaging faults sit:\n")
    rows, est, est_reps = [], 0.0, np.zeros(N_BOOT)
    for g, u in g_units.items():
        wgt = gsize[g] / total
        est += wgt * u.share("dmg")
        if not u.exact:
            est_reps += wgt * u.reps("dmg")
    for g, u in g_units.items():
        wgt = gsize[g] / total
        rows.append([g, f"{100 * wgt:.2f}", u.fmt("dmg", digits=2), f"{100 * wgt * u.share('dmg') / est:.1f}"])
    table(["group", "% of network's weights", "damaging % of its faults", "% of the network's damaging faults"],
          rows)
    exact_all = all(u.exact for u in g_units.values())
    ci = "" if exact_all else (f" [{np.percentile(est_reps, 2.5) * 100:.2f}, "
                               f"{np.percentile(est_reps, 97.5) * 100:.2f}]")
    print(f"Network-wide damaging share (size-weighted): {100 * est:.2f}%{ci}"
          f"{' (exact)' if exact_all else ''}, i.e. about {est * total * 16:,.0f} of the "
          f"{total * 16:,} possible single-bit weight faults.\n")

    # ------------------------------------------------ Q3: why fc stands out
    print("## Q3. Where fc stands out, and why\n")
    print("Q3a. fc against the other groups on each measure.\n")
    rows = []
    for g, u in g_units.items():
        r = u.rows
        b14 = r[d["bit"][r] == 14]
        big = r[d["large"][r]]
        rows.append([g, u.fmt("sdc"), u.fmt("large"), u.fmt("nonfin"), u.fmt("dmg"),
                     f"{u.count('dmg14')} of {len(b14)}", drop_summary(d["drop"][big]),
                     drop_summary(d["drop"][b14]), f"{d['acc'][b14].min():.4f}" if len(b14) else "–"])
    table(["group", "SDC %", "large SDC %", "non-finite %", "damaging %", "bit-14 damaging",
           "median drop of large SDC (pp)", "median bit-14 drop (pp)", "worst bit-14 accuracy"], rows)
    b14 = d["bit"] == 14
    print(f"All bit-14 faults: median drop {np.median(d['drop'][b14]):.2f} pp (non-finite faults included at their "
          f"logged accuracy).\n")

    print("Q3b. Bit-14 outcomes, binned by accuracy drop.\n")
    rows = []
    for g, u in g_units.items():
        r = u.rows[d["bit"][u.rows] == 14]
        rows.append([g, len(r)] + bin_counts(d, r))
    table(["group", "bit-14 faults"] + BIN_HEADER, rows)

    print("Q3c. Weight magnitudes (all weights in the FP16 model) and the bit-14 regime they fall in. "
          "|w| < 1: ×65,536. 1 ≤ |w| < 2: exponent all ones → inf/NaN. |w| ≥ 2: bit 14 already set → shrinks.\n")
    rows = []
    for g in group_order:
        a = np.concatenate([model.get_parameter(t).detach().half().abs().float().numpy().ravel() for t in grp[g]])
        rows.append([g, f"{a.size:,}", f"{np.median(a):.4f}", f"{np.percentile(a, 99):.3f}", f"{a.max():.3f}",
                     int(((a >= 1) & (a < 2)).sum()), int((a >= 2).sum())])
    for t in ("conv1.weight",):
        a = model.get_parameter(t).detach().half().abs().float().numpy().ravel()
        rows.append([f"(of which stem {t})", f"{a.size:,}", f"{np.median(a):.4f}", f"{np.percentile(a, 99):.3f}",
                     f"{a.max():.3f}", int(((a >= 1) & (a < 2)).sum()), int((a >= 2).sum())])
    table(["group", "weights", "median |w|", "99th pct |w|", "max |w|", "1 ≤ |w| < 2", "|w| ≥ 2"], rows)
    print("Bit-14 faults by |w| band and sign of w (faulted weights only).\n")
    rows = []
    aw = np.abs(d["w"])
    for g, u in g_units.items():
        r = u.rows[d["bit"][u.rows] == 14]
        for band, m in (("|w| < 1", aw[r] < 1), ("1 ≤ |w| < 2", (aw[r] >= 1) & (aw[r] < 2)), ("|w| ≥ 2", aw[r] >= 2)):
            for sgn, ms in (("w > 0", d["w"][r] > 0), ("w < 0", d["w"][r] < 0)):
                rr = r[m & ms]
                if len(rr):
                    rows.append([g, band, sgn, len(rr), int(d["dmg"][rr].sum()), int((d["cls"][rr] == 2).sum()),
                                 f"{np.median(d['acc'][rr]):.4f}"])
    table(["group", "band", "sign", "faults", "damaging", "non-finite", "median accuracy"], rows)

    conv_rule(d, model, g_units, tnames)
    if torch.cuda.is_available():
        fc_check(d, info, model, tnames)
    else:
        print("Q3e skipped: CUDA not available.\n")
    if fig_dir:
        figure(d, model, g_units, tnames, label, fig_dir, info["dataset"])


# ---------------------------------------------------------------- Q3d: conv sign rule

def bn_of(conv_name):
    """conv1.weight -> bn1.weight; layer2.0.conv2.weight -> layer2.0.bn2.weight."""
    parts = conv_name.split(".")
    parts[-2] = parts[-2].replace("conv", "bn")
    return ".".join(parts)


def conv_up(d, model, tnames, rows):
    """For conv faults: True when sign(w) * sign(gamma of the following BN channel) > 0."""
    out = np.zeros(len(rows), dtype=bool)
    for i, t in enumerate(tnames):
        if t == "fc.weight":
            continue
        m = d["t"][rows] == i
        if not m.any():
            continue
        wshape = model.get_parameter(t).shape
        gamma = model.get_parameter(bn_of(t)).detach().numpy()
        ch = d["e"][rows][m] // (wshape[1] * wshape[2] * wshape[3])
        out[m] = np.sign(d["w"][rows][m]) * np.sign(gamma[ch]) > 0
    return out


def conv_rule(d, model, g_units, tnames):
    print("Q3d. Conv bit-14 faults: does sign(w) × sign(BN γ) predict damage? Every conv except the stem reads "
          "ReLU outputs (≥ 0), so ×65,536 pushes its output channel hugely in the direction of sign(w); the next "
          "batch norm multiplies by γ/√var. Prediction: 'up' (hugely positive after BN, passes the next ReLU) → "
          "damaging; 'down' (hugely negative, zeroed by ReLU) → mostly harmless. The stem reads normalized pixels, "
          "which can be negative, so the rule is not expected to hold there.\n")
    stem = tnames.index("conv1.weight") if "conv1.weight" in tnames else -1
    rows = []
    for g, u in g_units.items():
        if g == "fc":
            continue
        r = u.rows[d["bit"][u.rows] == 14]
        for part, rr in (("stem conv1", r[d["t"][r] == stem]), ("rest", r[d["t"][r] != stem])):
            if not len(rr):
                continue
            up = conv_up(d, model, tnames, rr)
            dm = d["dmg"][rr]
            nf = d["cls"][rr] == 2
            rows.append([g, part, len(rr), f"{int((up & dm).sum())} of {int(up.sum())}",
                         f"{int((~up & dm).sum())} of {int((~up).sum())}", int((up & nf).sum()), int((~up & nf).sum()),
                         drop_summary(d["drop"][rr[up]]), drop_summary(d["drop"][rr[~up]]),
                         f"{int((up == dm).sum())} of {len(rr)}"])
    table(["group", "tensors", "bit-14 faults", "'up': damaging", "'down': damaging", "'up': non-finite",
           "'down': non-finite", "'up': median drop (pp)", "'down': median drop (pp)", "rule agrees"], rows)
    print("The same conv bit-14 faults (stem excluded), binned by accuracy drop.\n")
    rows = []
    for g, u in g_units.items():
        if g == "fc":
            continue
        r = u.rows[(d["bit"][u.rows] == 14) & (d["t"][u.rows] != stem)]
        up = conv_up(d, model, tnames, r)
        for name, m in (("'up'", up), ("'down'", ~up)):
            rows.append([g, name, int(m.sum())] + bin_counts(d, r[m]))
    table(["group", "prediction", "faults"] + BIN_HEADER, rows)


# ---------------------------------------------------------------- Q3e: fc check on the GPU

@torch.no_grad()
def fc_inputs(info, model):
    """FP16 inputs to fc (pooled features), FP16 logits and labels for all test images, as in the campaign."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _, _, x_test, y_test, raw = load_data(info["dataset"])
    require(raw == info["raw_sha256"], "dataset files differ from the run")
    m16 = copy.deepcopy(model).half().to("cuda")
    x16 = normalize(x_test.to("cuda"), info["normalize_mean"], info["normalize_std"]).half()
    got = {}
    hook = m16.fc.register_forward_hook(lambda mod, inp, out: got.__setitem__("a", inp[0].cpu()))
    logits = evaluate(m16, x16)
    hook.remove()
    if "reference_logits_sha256" in info:
        require(sha256_bytes(logits.numpy().tobytes()) == info["reference_logits_sha256"],
                "clean logits differ from the run's reference")
    del m16, x16
    torch.cuda.empty_cache()
    return got["a"].double().numpy(), logits.double().numpy(), y_test.numpy()


def fc_check(d, info, model, tnames):
    if "fc.weight" not in tnames:
        return None
    a, logits, y = fc_inputs(info, model)
    n_img = len(y)
    ref_pred = logits.argmax(1)
    print("Q3e. fc bit-14 faults predicted from the clean network alone. fc's inputs are averages of ReLU outputs, "
          f"so all ≥ 0 (min here {a.min():.3g}). A fault on fc.weight[c, j] adds (w_new − w)·a_j to class c's score "
          "and to nothing else. Predicted class: non-finite if w_new is inf/NaN or any new score reaches ±65,520 "
          "(FP16 overflow); otherwise SDC or masked from the new argmax. Predicted accuracy from the new argmax.\n")
    frac_pos = (a > 0).mean(0)
    amax = a.max(0)
    print(f"Pooled features: share of images with a_j > 0, median over the {a.shape[1]} features "
          f"{np.median(frac_pos):.3f} (min {frac_pos.min():.3f}); per-feature maximum over images: median "
          f"{np.median(amax):.2f}, range {amax.min():.2f}–{amax.max():.2f}. A finite bit-14 weight overflows a "
          f"score when 65,536·|w|·a_j ≳ 65,504, i.e. |w|·a_j ≳ 1.\n")
    ti = tnames.index("fc.weight")
    k = a.shape[1]
    correct_c = np.bincount(y[ref_pred == y], minlength=logits.shape[1])
    inv = {v: k_ for k_, v in CLASS_CODE.items()}

    def predict(r):
        c, j = divmod(int(d["e"][r]), k)
        w, wn = float(d["w"][r]), float(d["w_new"][r])
        if not np.isfinite(wn):
            return "non_finite", np.nan, c, j, w, "weight NaN/inf"
        lc = logits[:, c] + (wn - w) * a[:, j]
        over = np.abs(lc) >= FP16_INF_AT
        new = logits.copy()
        new[:, c] = np.where(over, np.copysign(np.inf, lc), lc)
        pred = new.argmax(1)
        cls = "non_finite" if over.any() else ("sdc" if (pred != ref_pred).any() else "masked")
        return cls, (pred == y).mean(), c, j, w, "overflow" if over.any() else "finite"

    all_rows = np.flatnonzero(d["t"] == ti)
    all_res = {int(r): predict(r) for r in all_rows}
    print("The same formula applied to every fc fault (all 16 bits), against the logged class and accuracy:\n")
    trows = []
    for name, lo, hi in (("mantissa (0–9)", 0, 9), ("exponent 10–13", 10, 13), ("exponent 14", 14, 14),
                         ("sign (15)", 15, 15)):
        rr = all_rows[(d["bit"][all_rows] >= lo) & (d["bit"][all_rows] <= hi)]
        ok = sum(all_res[int(r)][0] == inv[int(d["cls"][r])] for r in rr)
        acc_ok = sum(abs(all_res[int(r)][1] - d["acc"][r]) <= 1e-4 + 1e-7 for r in rr if np.isfinite(all_res[int(r)][1]))
        n_fin = sum(np.isfinite(all_res[int(r)][1]) for r in rr)
        trows.append([name, len(rr), f"{ok} of {len(rr)}", f"{acc_ok} of {n_fin}",
                      int(d["dmg"][rr].sum()), sum(all_res[int(r)][0] == "non_finite" for r in rr)])
    table(["bits", "fc faults", "class predicted correctly", "accuracy within one image", "damaging (logged)",
           "non-finite (predicted)"], trows)

    print("Bit 14 in detail.\n")
    rows = all_rows[d["bit"][all_rows] == 14]
    res = [all_res[int(r)] for r in rows]
    logged_cls = d["cls"][rows]
    agree_cls = sum(rc[0] == inv[int(lc)] for rc, lc in zip(res, logged_cls))
    acc_rows = [(rc[1], float(d["acc"][r])) for rc, r in zip(res, rows) if np.isfinite(rc[1])]
    diffs = np.array([abs(p - o) for p, o in acc_rows]) * 100
    print(f"Class predicted correctly: {agree_cls} of {len(rows)}. Accuracy predicted (faults with a finite new "
          f"weight, {len(acc_rows)}): within 0.01 pp (one image) for {int((diffs <= 0.01 + 1e-9).sum())}, within 0.1 pp "
          f"for {int((diffs <= 0.1 + 1e-9).sum())}; largest miss {diffs.max():.2f} pp.\n")
    routes = {}
    for rc, lc in zip(res, logged_cls):
        routes.setdefault(rc[5], []).append(inv[int(lc)])
    print("How fc's bit-14 faults became non-finite (route predicted from w and the features, class as logged):\n")
    table(["predicted route", "faults", "logged non-finite", "logged SDC", "logged masked"],
          [[k_, len(v), v.count("non_finite"), v.count("sdc"), v.count("masked")] for k_, v in routes.items()])
    print("Rule of thumb, sign of w (fc bit-14 faults with finite new weight): w > 0 → class c wins for (almost) every "
          "image with a_j > 0 → accuracy near 10%; w < 0 → class c is (almost) never predicted → accuracy falls by "
          "class c's share of correct answers.\n")
    rows_t = []
    for sgn in (1, -1):
        sel = [(rc, r) for rc, r in zip(res, rows) if np.sign(rc[4]) == sgn and rc[5] != "weight NaN/inf"]
        if not sel:
            continue
        acc = np.array([float(d["acc"][r]) for _, r in sel])
        if sgn > 0:
            rule = np.array([(y == rc[2]).mean() for rc, _ in sel])
        else:
            rule = np.array([info["fp16_cuda_accuracy"] - correct_c[rc[2]] / n_img for rc, _ in sel])
        err = np.abs(acc - rule) * 100
        rows_t.append(["w > 0" if sgn > 0 else "w < 0", len(sel), f"{np.median(acc):.4f}",
                       f"{acc.min():.4f}–{acc.max():.4f}", "share of images labelled c" if sgn > 0 else
                       "ref − (correct images of class c)", f"{np.median(err):.2f}", f"{err.max():.2f}"])
    table(["sign", "faults", "median accuracy", "range", "rule", "median |acc − rule| (pp)",
           "max |acc − rule| (pp)"], rows_t)
    odd = [(rc, r) for rc, r in zip(res, rows) if rc[4] > 0 and rc[5] != "weight NaN/inf" and d["acc"][r] > 0.15]
    for rc, r in odd:
        print(f"- w > 0 exception: fc.weight[{rc[2]}, {rc[3]}] w = {rc[4]:.3g}, max a_j = {a[:, rc[3]].max():.2f}, "
              f"so the added score is at most {(float(d['w_new'][r]) - rc[4]) * a[:, rc[3]].max():.1f}; "
              f"accuracy {float(d['acc'][r]):.4f}.")
    if odd:
        print()


# ---------------------------------------------------------------- figure

def figure(d, model, g_units, tnames, label, fig_dir, ds):
    """Accuracy of every bit-14 fault, one column per group, marked by the sign rule's prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#2b2b29", "#6b6a64", "#e5e5e3", "#fcfcfb"
    series = (("predicted 'up': output pushed hugely positive", "#2a78d6", "^"),
              ("predicted 'down': output pushed hugely negative", "#eb6834", "v"),
              ("non-finite output (inf or NaN)", "#1baf7a", "X"))
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    rng = np.random.default_rng(1)
    names = list(g_units)
    for x, g in enumerate(names):
        r = g_units[g].rows[d["bit"][g_units[g].rows] == 14]
        up = d["w"][r] > 0 if g == "fc" else conv_up(d, model, tnames, r)
        nf = d["cls"][r] == 2
        acc = 100 * d["acc"][r]
        jit = x + rng.uniform(-0.3, 0.3, len(r))
        for m, (lab, col, mk) in zip((up & ~nf, ~up & ~nf, nf), series):
            ax.scatter(jit[m], acc[m], s=22, color=col, marker=mk, edgecolors=surface, linewidths=0.5,
                       label=lab if x == 0 else None, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["stem + stage 1", "stage 2", "stage 3", "final layer (fc)"][:len(names)], color=ink)
    ax.set_ylabel("test accuracy with the fault (%)", color=ink)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(grid)
    ax.set_title(f"{label}: accuracy after each bit-14 fault (150 sampled weights per group)\n"
                 "conv prediction = sign(w) × sign(batch-norm γ); fc prediction = sign(w)",
                 fontsize=10, loc="left", color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, labelcolor=ink)
    fig.tight_layout()
    out = Path(fig_dir) / f"damage_bit14_{'fmnist' if ds == 'fashion_mnist' else ds}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("faults", nargs="+", help="faults.csv files (run.json must sit next to each)")
    ap.add_argument("--figures", help="folder for the bit-14 figure (one PNG per file); omit for no figures")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for p in args.faults:
        analyze(Path(p).resolve(), args.figures)


if __name__ == "__main__":
    main()
