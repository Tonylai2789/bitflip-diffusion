# Plan and work log: ResNet-20 bit-flip study (Week 2, Nov 10 – Nov 16, 2026)

**Audience:** AI coding agents (and people) continuing this work. Written to be read top to bottom with no other context.
**Human-readable summary:** [`README.md`](README.md) (in this folder).
**Paths:** every path in this file is relative to `resnet20/`, and commands run from there, unless the path starts with `../`. This folder has its **own** virtual environment, `resnet20/.venv` (the repo root's `.venv` belongs to `../cifar10/`). `../mnist/` is the Week 1 study this one copies from.
**Repo:** `Tonylai2789/bitflip-diffusion`. This plan was written at commit `5c79b5a`.
**Owner:** Tony Lai. Team: Aneesh. Mentor: Parnian Shabani Kamran. PI: Prof. Hussain Al-Asaad (UC Davis).
**Due:** Nov 16, 2026 (end of the Week 2 window in the doc).
**Status (Nov 18):** Phase 1 done. Decisions D1–D9 settled Nov 17. `src/resnet20_fault.py` written and tested; a full **development** pipeline on Fashion-MNIST passed every check (section 6). Code committed and pushed Nov 18 (`27ee894`). **Official runs done Nov 18** for both datasets from the clean tree: all checks passed (work log). Results write-up committed Nov 21 (`9a82d79`): [`results-2026-11-20.md`](results-2026-11-20.md). Drafts committed Nov 21: [`for-vgg-group-2026-11-21.md`](for-vgg-group-2026-11-21.md) (our steps and results for the VGG group to compare) and [`answers-draft-2026-11-21.md`](answers-draft-2026-11-21.md) (fault-model paragraph and short answers, for Tony to rewrite). Next: the VGG comparison, once their results arrive (section 6 step 6). See the [work log](#9-work-log).

---

## 0. Read this first (rules for any agent working here)

1. **Order of authority.** (1) The team Google Doc *Resilience of Diffusion Models Under Bit Flip Faults*: the **Tasks → Week2** tab (this week's task) and the **Tasks → Week1** tab (the fault model and classification rules this week reuses). The **Proposal** tab is the overall source of truth. (2) This file. (3) `../mnist/plan-mnist-week1.md` for conventions. Where they disagree, the higher one wins. **Never edit the Google Doc.**
2. **Open conflict in the doc.** The Proposal tab's "Tasks (Nov 9 – Nov 16)" section describes a diffusion fault injector (Tier S), while the Week2 tab describes this classifier grid. Tony is confirming with Parnian which applies. Until then this plan follows the Week2 tab.
3. **Tony commits and pushes.** Agents prepare files, run checks and give Tony exact commands. Do not commit, push or create repos unless Tony asks in that session. Commits on this PC pass Tony's identity with `git -c user.name=... -c user.email=...`; do not change git config.
4. **The fault model is the Week 1 fault model, adapted only where ResNet-20 forces it** (section 3). Any deviation from the doc needs a written reason in the README and in that run's `run.json`, and Tony's approval first.
5. **Decisions in section 4 marked "Proposed" are not settled.** No official run until every one is "Settled".
6. **Tony writes anything the doc asks for "in your own words"** (paragraphs, short answers). Agents supply numbers and check facts.
7. **Record the git commit hash (and dirty flag) with every result.** Development runs first; official results only from a clean, committed tree.
8. **One results machine and stack:** the Windows RTX 5090 PC with the stack pinned in `requirements.txt` (this folder). Never mix in Mac runs.
9. **Keep this file current.** After every step (pass or fail), append a row to the [work log](#9-work-log) and update the step's status.
10. **Stop and ask** on any failed check. Do not work around it silently.
11. Before moving or creating files, check `git status` and whether another Claude session is working in this checkout (two sessions broke each other's paths on Nov 8).
12. Tony's global `~/.claude/CLAUDE.md` (Karpathy guidelines) applies: minimum code, no speculative features, surgical changes, verify each step.

## 1. Goal and scope

**Research goal (Proposal tab):** find which parts of a diffusion image generator turn single-bit FP16 weight faults into silent corruption (RQ1), whether more denoising steps absorb transient faults (RQ2), and whether small-model conclusions hold at Stable Diffusion scale (RQ3). The study climbs a ladder: MLP (Week 1) → small CNNs (this week) → CIFAR-10 DDPM (Tier S) → compressed and full Stable Diffusion (Tiers M, L).

**This week (Week2 tab):** a 2 × 2 grid of {small VGG-style network, ResNet-20} × {Fashion-MNIST, CIFAR-10}. **Our group does the ResNet-20 row** (both datasets, pending confirmation). The VGG row is another group's.

**Why ResNet-20 matters for the study:** it has convolutions, skip connections and batch normalization, like the diffusion UNet. Comparing it with the plain VGG shows whether skip connections and normalization absorb or spread a bit flip; comparing datasets shows whether 1- vs 3-channel inputs change the picture.

**Out of scope (Week1 tab, carried over):** faults in activations, inputs, biases or arithmetic hardware; more than one bit at once.

## 2. Model and data

**ResNet-20** (He et al. 2016, CIFAR variant): a 3 × 3 convolution with 16 filters (the stem), then 3 stages of 3 basic blocks each (16, 32, 64 filters; each block is conv–BN–ReLU–conv–BN plus a shortcut, then ReLU), global average pool, linear layer 64 → 10. Stages 2 and 3 halve the spatial size with stride 2. Shortcuts are option A (identity with zero-padding, no weights), so the network has **20 weight tensors**: 19 convolutions + 1 linear.

Weight counts (by hand; to be checked against the model code):

| Group | Tensors | CIFAR-10 (3 channels) | Fashion-MNIST (1 channel) |
|---|---|---|---|
| Stem conv | 1 | 432 | **144** |
| Stage 1 convs | 6 | 13,824 | 13,824 |
| Stage 2 convs | 6 | 50,688 | 50,688 |
| Stage 3 convs | 6 | 202,752 | 202,752 |
| Linear (`fc.weight`) | 1 | 640 | 640 |
| **Total weights** | 20 | 268,336 | 268,048 |

Plus 10 linear biases and 19 batch-norm layers (scale, shift, running mean and variance).

**Data** (downloaded with torchvision into `data/`, git-ignored):
- CIFAR-10: 50,000 train / 10,000 test, 3 × 32 × 32.
- Fashion-MNIST: 60,000 train / 10,000 test, 1 × 28 × 28. ResNet-20 runs on 28 × 28 directly (28 → 14 → 7 through the stages).

The doc's CIFAR-10 link (`cave.cs.toronto.edu/kriz/cifar.html`) looks like a typo for `www.cs.toronto.edu/~kriz/cifar.html`. We use torchvision's download, which checks the archive's MD5.

## 3. Fault model (Week1 tab rules, adapted)

Unchanged from Week 1: one bit of one stored FP16 weight inverted; all 16 bits tested on every chosen element; the flipped value stays for the whole evaluation; restore by writing the saved pattern back, verified by SHA-256; evaluate the same fixed set every time (all 10,000 test images of that dataset); reference = the same trained network with no fault, in FP16.

**Classification** (in order; exactly one class each): `non_finite` (any logit inf or NaN) → `sdc` (all finite, at least one prediction differs from the fault-free FP16 prediction) → `masked`.

**Adaptations ResNet-20 forces** (each needs Tony's approval and a written reason):
- **Sampling groups.** Week 1 drew 150 elements from each of its two matrices. ResNet-20 has 20 weight tensors, and the Fashion-MNIST stem has only 144 weights, so "150 per tensor" is impossible there. See decision D2.
- **What counts as a weight.** Week 1 excluded biases. ResNet-20 also has batch-norm parameters and statistics. See decision D3.

## 4. Decisions

| # | Decision | Choice | Status |
|---|---|---|---|
| D1 | Which datasets our group runs | Both (the whole ResNet-20 row) | Settled (Tony, Nov 17) |
| D2 | Sampling groups | 4 groups: stem + stage 1, stage 2, stage 3, `fc.weight`. 150 distinct elements per group, uniform without replacement → 600 elements × 16 bits = **9,600 faults per dataset** | Settled (Tony, Nov 17) |
| D3 | Batch-norm parameters and statistics | Excluded, like biases; only conv and linear weights are faulted | Settled (Tony, Nov 17) |
| D4 | Master seed and streams | Master seed `20261104` (Week 1's), with new numpy `SeedSequence(seed, spawn_key=(role,))` roles for this study (see `configs/resnet20.json`); training and sampling streams separate, per the Week2 tab | Settled (Tony, Nov 17) |
| D5 | Training recipe | SGD, momentum 0.9, lr 0.1 divided by 10 at epochs 80 and 120, 160 epochs, weight decay 1e-4, batch 128, cross-entropy (He et al.). CIFAR-10: random crop (pad 4) + horizontal flip. Fashion-MNIST: no augmentation. Train in FP32 on CUDA. To align with the VGG group. | Settled (Tony, Nov 17) |
| D6 | Normalization | Per-channel mean and std computed from each training set, recorded in `run.json` | Settled (Tony, Nov 17) |
| D7 | FP16 accuracy margin | FP16 test accuracy within 0.1 percentage points of FP32 (Week 1's margin, itself unconfirmed with Parnian) | Settled (Tony, Nov 17) |
| D8 | Evaluation | Whole network cast to FP16 (`model.half()`), on CUDA, `torch.backends.cudnn.deterministic = True`, `benchmark = False`; campaign run twice, result files must be byte-identical | Settled (Tony, Nov 17) |
| D9 | Deliverables | Same shape as Week 1: per-bit and per-field tables with Wilson 95% intervals, a plot, Tony's fault-model paragraph and short answers; plus Fashion-MNIST vs CIFAR-10 and ResNet vs VGG comparisons | Settled (Tony, Nov 17) |

## 5. Phase 1: setup and understanding

| Step | Action | Check | Status |
|---|---|---|---|
| 1 | `git pull --ff-only` | local `main` = `origin/main` | Done |
| 2 | Create `resnet20/` (`src/`, `tests/`, `configs/`, `logs/`), README, this plan; one line in root README, `CLAUDE.md`, `AGENTS.md` | `git status` shows only these | Done |
| 3 | `py -3.12 -m venv .venv`; install `torch torchvision` from the cu134 index, then `numpy pytest`; pin with `cmd /c ".venv\Scripts\python.exe -m pip freeze > requirements.txt"` (PowerShell 5.1's `>` writes UTF-16) | smoke test prints `True` and `NVIDIA GeForce RTX 5090` | Done |
| 4 | Download CIFAR-10 and Fashion-MNIST with torchvision into `data/` | 50,000/10,000 and 60,000/10,000 images | Done |
| 5 | Copy `../mnist/src/bitflip.py` (unchanged) and `../mnist/tests/test_bitflip.py` (GPU device MPS → CUDA) | all tests pass, CUDA cases not skipped | Done |
| 6 | `configs/resnet20.json` with the section 4 decisions | file matches section 4 | Done |
| 7 | Tony reviews and commits | | Done (Nov 18) |

To recreate the environment: `py -3.12 -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu134`.

## 6. Later phases (after section 4 is settled)

Per dataset (Fashion-MNIST first: smaller, faster). Steps 1–5 done as **development** runs on Fashion-MNIST (Nov 17, dirty tree, `logs/dev_fmnist_*`); see the work log. Steps 2–5 done as **official** runs on both datasets (Nov 18, clean tree `27ee894`, `logs/fmnist_*` and `logs/cifar10_*`).
1. **Model code** in `src/`: ResNet-20 and a CLI like `../mnist/src/mnist_fault.py` (`train`, `one-weight`, `campaign`, `summarize`, `compare`). Check: weight counts match section 2.
2. **Train** once in FP32 on CUDA; save weights and their SHA-256; record dataset, loader, versions, normalization, FP32 and FP16 accuracy. Check: FP16 within the D7 margin. Training is not bit-repeatable on GPU (Week2 tab), so it runs once and every campaign loads the saved file.
3. **One-weight walk:** one element through all 16 bits, by hand-checkable values.
4. **Campaign** twice (`campaign_a`, `campaign_b`); check byte-identical `faults.csv`.
5. **Summarize:** per-bit and per-field tables, Wilson intervals, plot.
6. **Compare:** Fashion-MNIST vs CIFAR-10; ResNet-20 vs the VGG group's results; vs Week 1's MLP.

### 6.1 Official run commands (from `resnet20/`, clean committed tree)

```
.venv\Scripts\python.exe src/resnet20_fault.py train --dataset fashion_mnist --run-name fmnist_train
.venv\Scripts\python.exe src/resnet20_fault.py one-weight --model logs/fmnist_train/model_fp32.pt --run-name fmnist_one_weight
.venv\Scripts\python.exe src/resnet20_fault.py campaign --model logs/fmnist_train/model_fp32.pt --run-name fmnist_campaign_a
.venv\Scripts\python.exe src/resnet20_fault.py campaign --model logs/fmnist_train/model_fp32.pt --run-name fmnist_campaign_b
.venv\Scripts\python.exe src/resnet20_fault.py compare logs/fmnist_campaign_a logs/fmnist_campaign_b
.venv\Scripts\python.exe src/resnet20_fault.py summarize logs/fmnist_campaign_a
```

Then the same with `--dataset cifar10` and `cifar10_` run names. Times on the 5090 (Fashion-MNIST dev run): training 6.2 min, each campaign 4.2 min. CIFAR-10 will be somewhat slower (larger images).

### 6.2 Things to know about the results

- **Report FP32 accuracy from the campaign's `run.json`, not the training one.** Training enables `cudnn.benchmark` (fastest convolution algorithms); evaluation forces deterministic ones. The tiny numeric difference moved 2 of 10,000 Fashion-MNIST images in the dev run (FP32 93.00% at training vs 92.98% in evaluation, same weights).
- **Near-tie images make low bits count as SDC.** In the dev model, one Fashion-MNIST test image (#4019) has its top two FP16 logits exactly equal (5.2578125), 3 are within 0.01 and 27 within 0.1. Any perturbation, even of mantissa bit 0, can flip such an image, and one changed image is SDC by the doc's rule. So bit 0 gave 12% SDC, almost all moving accuracy by ±1 image, whereas Week 1's MLP had bits 0–2 always masked. This follows the doc's definition exactly; whether to also report the size of the change (number of changed predictions) is a question for Parnian (section 7).

## 7. Open questions (for Tony / Parnian)

Settled Nov 17: our group runs the whole ResNet-20 row (both datasets); D1–D9 accepted as proposed.

Still open:
1. Which task is this week's: the Week2 tab grid, or the Proposal tab's "Tasks (Nov 9 – Nov 16)" diffusion injector? (Proceeding with the Week2 grid.)
2. The deadline: the Week2 window ended Nov 16.
3. Tell the VGG group our recipe (D5) and normalization (D6), so the grid is comparable; if theirs differs, record the difference.
4. Parnian's confirmation of the deviations D2 (sampling groups) and D3 (batch norm excluded), and of the 0.1 pp FP16 margin (D7, carried over from Week 1).
5. SDC counts a fault if even one of 10,000 predictions changes, so near-tie images make tiny faults count (section 6.2). Keep the rule as is (it is the doc's), and should `faults.csv` also record how many predictions changed? (Currently it records accuracy only, like Week 1.)

## 8. Environment

Windows 11 Pro (DESKTOP-0TAKR5L), RTX 5090 32 GB, NVIDIA driver 621.01, Python 3.12.10. Exact packages in `requirements.txt`.

## 9. Work log

Append one row per step or significant action. Newest last.

| Date | Who | Action | Result | Commit |
|---|---|---|---|---|
| 2026-11-15 | Claude (PC) | Read the Google Doc (Week2, Week1, Proposal tabs) and the Week 1 repo work; drafted the Phase 1 plan | Week2 tab gives the grid and reproducibility rules only; fault model comes from Week1. Conflict with Proposal "Tasks (Nov 9 – Nov 16)" noted. Fashion-MNIST stem has 144 weights (< 150). Aryan and Jai's repo not accessible. | `c611a20` |
| 2026-11-16 | Claude (PC) | Step 1: `git pull --ff-only` | Fast-forward `c611a20` → `5c79b5a`; HEAD = origin/main | `5c79b5a` |
| 2026-11-16 | Claude (PC) | Step 2: created `resnet20/` and this plan; added one sentence to root `CLAUDE.md`, `AGENTS.md`, README layout | Done | `5c79b5a` (uncommitted) |
| 2026-11-16 | Claude (PC) | Step 3: `resnet20/.venv` (Python 3.12.10); `torch torchvision` from cu134 index, then `numpy pytest`; pinned `requirements.txt` via `cmd /c` | torch 2.15.0+cu134, torchvision 0.29.2+cu134, numpy 2.5.4, pytest 9.2.0. Smoke test: `True`, `NVIDIA GeForce RTX 5090`. `requirements.txt` is ASCII. | `5c79b5a` (uncommitted) |
| 2026-11-16 | Claude (PC) | Step 4: torchvision download of CIFAR-10 and Fashion-MNIST into `data/` | CIFAR-10 50,000/10,000 RGB 32×32; Fashion-MNIST 60,000/10,000 L 28×28. `data/` = 423 MB, ignored by root `.gitignore`. CIFAR-10 server was slow (~15 min). | `5c79b5a` (uncommitted) |
| 2026-11-16 | Claude (PC) | Step 5: copied `bitflip.py` (byte-identical) and `test_bitflip.py` (GPU device MPS → CUDA); ran pytest | 47 passed, 0 skipped (CPU and CUDA), including hand values 0.5 bit 14 → 32768, 1.0 bit 14 → inf, 0.5 bit 0 → 0.50048828125, 0.5 bit 15 → −0.5 | `5c79b5a` (uncommitted) |
| 2026-11-16 | Claude (PC) | Step 6: wrote `configs/resnet20.json` (marked PROPOSED) and `README.md` | JSON parses; values match section 4 | `5c79b5a` (uncommitted) |
| 2026-11-17 | Claude (PC) | Went over D1–D9 with Tony | All accepted as proposed: both datasets, 4 sampling groups × 150 (9,600 faults per dataset), batch norm excluded, master seed 20261104 with roles 10–16, 160-epoch He et al. recipe, normalization from training sets, 0.1 pp FP16 margin, FP16 deterministic CUDA eval run twice, Week 1-style deliverables. Updated section 4, section 7, config, README. | `5c79b5a` (uncommitted) |
| 2026-11-17 | Claude (PC) | Section 6 step 1: wrote `src/resnet20_fault.py` (model, train, one-weight, campaign, summarize, compare; structure from `../mnist/src/mnist_fault.py`); added `tests/test_model.py`, copied `tests/test_wilson.py`; installed matplotlib and re-pinned `requirements.txt` | 52 tests pass. Weight counts match section 2 exactly (268,048 / 268,336 weights; 269,434 / 269,722 parameters). Data held as GPU tensors (no DataLoader); augmentation on GPU, randomness from the training seed. | `5c79b5a` (uncommitted) |
| 2026-11-17 | Claude (PC) | **Dev** train, Fashion-MNIST (`logs/dev_fmnist_train`) | 160 epochs, 6.2 min. FP32 93.00%, FP16 92.98% (cudnn.benchmark on). | `5c79b5a` dirty |
| 2026-11-17 | Claude (PC) | **Dev** one-weight (`logs/dev_fmnist_one_weight`) | `layer3.1.conv1.weight[29070]` = −0.07684326171875, first draw. Hand-checked bits 12, 13, 14 (−1.2294921875, −0.00030017, −5036.0). Eval: FP32 92.98%, FP16 92.98%, diff 0.0 pp, double eval identical; sanity fault (`fc.weight[85]` bit 14 → 62016) gave non-finite logits, restore verified. | `5c79b5a` dirty |
| 2026-11-17 | Claude (PC) | **Dev** campaign ×2, compare, summarize (`logs/dev_fmnist_campaign_a`, `_b`) | 9,600 faults each, 4.2 min each. masked 4,604 / sdc 4,931 / non_finite 65 (all non-finite at bit 14; 54 of them in `fc`). `chosen_elements.csv` and `faults.csv` byte-identical. SDC by group: stage1 83.5%, stage2 55.6%, stage3 30.7%, fc 35.6%. Low-bit SDC traced to near-tie images (section 6.2). | `5c79b5a` dirty |
| 2026-11-18 | Claude (PC) | Phase 1 step 7: committed and pushed (Tony's OK) root `CLAUDE.md`, `AGENTS.md`, `README.md` and `resnet20/` code, config, tests, README and this plan; dev logs left uncommitted | Pushed to `origin/main` | this commit |
| 2026-11-18 | Claude (PC) | **Official** Fashion-MNIST: train, one-weight, campaign ×2, compare, summarize (`logs/fmnist_*`) | Every `run.json`: commit `27ee894`, dirty false. Training 160 epochs, 6.2 min; normalization mean 0.2860, std 0.3530. Eval (deterministic): FP32 93.33%, FP16 93.32%, diff 0.01 pp (within D7). Sanity fault `fc.weight[219]` bit 14 → 65216: non-finite, restore verified. One weight `layer3.1.conv1.weight[29070]` = 0.060333251953125; hand-checked bit 14 → 3954, bit 12 → 0.96533203125. Campaign 9,600 faults, 4.2 min each: masked 5,017 / sdc 4,525 / non_finite 58 (all bit 14); both result files byte-identical. SDC by group: stage1 63.2%, stage2 59.7%, stage3 29.2%, fc 36.5%. Bit 0 SDC 16.0%. Training differs from the dev model (GPU training not bit-repeatable), as expected. | `27ee894` clean |
| 2026-11-18 | Claude (PC) | **Official** CIFAR-10: same six steps (`logs/cifar10_*`); first time the CIFAR-10 path ran | Every `run.json`: commit `27ee894`, dirty false. Training 160 epochs, 5.9 min; normalization mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616). Eval: FP32 91.38%, FP16 91.35%, diff 0.03 pp (within D7; He et al. report 91.25%). Sanity fault `fc.weight[200]` bit 14 → 65344: non-finite, restore verified. One weight `layer3.1.conv1.weight[28942]` = −0.1248779296875; hand-checked bit 14 → −8184. Campaign 9,600 faults, 5.3 min each: masked 3,300 / sdc 6,172 / non_finite 128 (all bit 14); both result files byte-identical. SDC by group: stage1 85.5%, stage2 75.5%, stage3 47.1%, fc 49.1%. Bit 0 SDC 24.2%. | `27ee894` clean |
| 2026-11-18 | Claude (PC) | Committed and pushed (Tony's OK) the official `fmnist_*` and `cifar10_*` logs (run.json, CSVs, per-bit plots, compare files; same set as Week 1), work-log rows and status lines | Model files (`*.pt`) and reference predictions (`*.npy`) stay ignored; dev logs uncommitted | this commit |
| 2026-11-20 | Claude (PC) | Wrote `results-2026-11-20.md` (results with figure guides) and `src/make_figures.py` (5 figures in `figures/` from the official logs) | Figure 5 re-evaluated both saved models: clean FP16 logits byte-identical to the campaign references. New numbers: 93% (F-MNIST) / 92% (CIFAR-10) of SDC moved accuracy ≤ 0.1 pp; bit 14 caused 93% / 79% of damaging faults; bits 0–2 SDC almost all in stem + stage 1 and stage 2; test images with top-2 gap < 0.1: 26 / 34. Committed and pushed with Tony's OK (figures force-added, as `*.png` is ignored). | this commit |
| 2026-11-21 | Claude (PC) | Drafted `for-vgg-group-2026-11-21.md` for the VGG group: our results (totals, by field, by bit, by group), every step of the method with settings and seeds, output file formats, the choices that decide comparability, questions for them and a fill-in comparison table. Updated the status line. | Every number checked against the official logs and `results-2026-11-20.md`. Committed and pushed with Tony's OK. | this commit |
| 2026-11-21 | Claude (PC) | Drafted `answers-draft-2026-11-21.md` for Tony to rewrite in his own words: the fault-model paragraph and the four short answers. The Week2 tab asks no new questions, so it answers the Week 1 questions (Proposal tab hand-in list) for both datasets. Read the Google Doc only; did not edit it | Every number checked against the official logs (`27ee894`) and `results-2026-11-20.md`; the per-bit table re-derived from `per_bit.csv`. New counts from `faults.csv` / `chosen_elements.csv`: making a weight bigger gave SDC more often than shrinking it (bit 12 ×16: 379/388 and 316/320); bit 14 turned the 4 / 5 `fc` weights ≥ 1 into NaN; every bits 0–2 SDC changed accuracy by ≤ 2 images net. Untested explanations labelled. Answer 2.3 (FP32) is reasoning only: no FP32 run. No VGG comparison. Committed and pushed with Tony's OK. | this commit |
| 2026-11-22 | Claude (PC) | Wrote `handoff-rtx4090-2026-11-22.md` for rerunning on a second machine with an RTX 4090: setup, smoke test, two ways to rerun (same models / from scratch) with commands and `rtx4090_` run names, checks, our reference numbers, what to do if results differ across GPUs. Force-added the official and dev models (`model_fp32.pt`, 1.1 MB each) and reference predictions so way A can load our exact weights; committed the dev logs. | Core code unchanged since `27ee894`. Measured peak GPU memory of a full test-set evaluation: 3.1 GB (F-MNIST), 4.1 GB (CIFAR-10). Installed torch has no `sm_89` kernels (runs via `sm_86`). Committed and pushed with Tony's OK. | this commit |
