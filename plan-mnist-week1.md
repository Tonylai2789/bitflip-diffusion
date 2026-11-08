# Plan and work log: MNIST bit-flip study (week of Nov 4 – Nov 9, 2026)

**Audience:** AI coding agents (and people) continuing this work. Written to be read top to bottom with no other context.
**Human-readable summary:** [`README-mnist-week1.md`](README-mnist-week1.md).
**Repo:** `Tonylai2789/bitflip-diffusion`. This plan was written at commit `2484ce4`.
**Owner:** Tony Lai. Team: Aneesh. Mentor: Parnian Shabani Kamran. PI: Prof. Hussain Al-Asaad (UC Davis).
**Due:** Monday, Nov 9, 2026.
**Status:** Planning done, all decisions settled. **No code written yet.** See the [work log](#14-work-log) for the latest state.

---

## 0. Read this first (rules for any agent working here)

1. **Order of authority.** (1) The "Proposal" tab of the team Google Doc *Resilience of Diffusion Models Under Bit Flip Faults*, section **Tasks (Nov 4 – Nov 9)**: https://docs.google.com/document/d/1q3cozz_Vtl_ZVSSwx1kGn-ljIXv3hj5XidglNhFjE9E/edit?tab=t.0. (2) This file. (3) [`handoff-mnist-week1.md`](handoff-mnist-week1.md), the earlier handoff from the dev-box session. Where they disagree, the higher one wins. **Never edit the Google Doc.**
2. **Tony commits and pushes.** Agents prepare files, run checks and give Tony exact commands. Do not commit, push or create repos unless Tony asks in that session.
3. **The fault model is fixed.** The doc says to use its rules exactly and to write down the reason for any change. Any deviation needs a written reason in the README and in that run's `run.json`, and Tony's approval first.
4. **The decisions in section 4 are settled.** Do not change them without asking Tony.
5. **Tony writes the step 7 paragraph and the four short answers himself** ("in your own words", per the doc). Agents supply numbers and check facts. Do not draft those texts, and do not write explanations of the results into the README or this file that would answer them.
6. **Record the git commit hash (and dirty flag) with every result.**
7. **Keep this file current.** After every step (pass or fail), append a row to the [work log](#14-work-log) and update the step's status line. Record what was run, the result, and anything surprising. A future agent must be able to resume from this file alone.
8. **Stop and ask** on any failed check listed under "Stop if" below. Do not work around it silently.
9. Tony's global `~/.claude/CLAUDE.md` (Karpathy guidelines) applies: minimum code, no speculative features, surgical changes, verify each step.

## 1. Goal and scope

**Goal (doc):** a working single-bit-flip fault injector on a small network, plus a full fault campaign with per-bit results. It is the warm-up for the diffusion-model study next week, and `src/bitflip.py` is reused there.

**In scope:** persistent single-bit faults in the two stored FP16 weight matrices of a trained MNIST MLP.

**Out of scope this week (doc):** faults in activations, inputs, biases or arithmetic hardware; more than one bit flipped at once; any diffusion work on the Mac.

**Separate result set.** These results come from Tony's MacBook (Apple M5 Pro, MPS). The diffusion work comes from the Windows RTX 5090 PC. Never mix Mac and PC runs in one result set, and say so in the README.

## 2. The fault model (from the doc, restated)

| Question | Rule |
|---|---|
| What is a fault? | Exactly one bit of one stored number is inverted (0 → 1 or 1 → 0). One fault per run (one run = one evaluation of the test set). |
| What is hit? | One element of one of the two weight matrices: `fc1.weight` (784 → 128) or `fc2.weight` (128 → 10), while stored in memory. Biases are excluded (138 of 101,770 parameters). |
| Number format | FP16: bit 15 sign, bits 14–10 exponent, bits 9–0 mantissa. Bit 0 is the rightmost (least significant) bit. |
| Which bit? | Not random. All 16 bits are tested on every chosen element. Report per bit, and grouped as sign / exponent / mantissa. |
| Which elements? | 150 distinct elements from each matrix, uniform, without replacement, from a seeded generator: 300 elements. The same 300 are used for all 16 bits. |
| Randomness | One master seed. Each purpose gets its own stream derived from it: training, the step-4 weight, the sampling of each matrix in step 5. Never reuse a stream for two purposes. |
| Duration | The flipped value stays for the whole evaluation. Afterwards the original bits are restored and verified against a SHA-256 fingerprint taken before the flip. |
| What is evaluated? | All 10,000 MNIST test images, the same for every fault. The reference is the same trained network with no fault. |

**Classification**, checked in this order; each fault gets exactly one class:

1. `non_finite`: any output logit is inf or NaN.
2. `sdc` (silent data corruption): all logits finite, and at least one of the 10,000 predictions differs from the fault-free prediction.
3. `masked`: all logits finite, and every prediction equals the fault-free prediction.

**CPU fallback (doc):** if FP16 can't run on a GPU, keep 32-bit storage and use 32 bits per element (bit 31 sign, 30–23 exponent, 22–0 mantissa). That gives 9,600 faults, and the report must say so. Do not switch to this, or to CPU FP16, without Tony's approval.

## 3. Where things stand

- The repo has `README.md` (diffusion project and research questions), `.gitignore`, `src/baseline.py` and `src/compare_runs.py` (DDPM clean baseline, run on the PC), empty `configs/` and `notebooks/`, and `logs/seed0_a`, `logs/seed0_b` (baseline runs, bitwise identical).
- `requirements.txt` is pinned for the Windows PC (`torch==2.15.0+cu134`) and will not install on a Mac. **Leave it alone.** The Mac gets `requirements-mac.txt`.
- No MNIST code exists yet.
- The handoff was checked against the Google Doc on Nov 7: no conflicts. The doc leaves the seed, stream method, preprocessing, training recipe, FP16 margin, training device and plot format open; Tony settled them on Nov 8 (section 4).

## 4. Settled decisions (Tony, Nov 8, 2026)

| # | Decision | Value | Notes |
|---|---|---|---|
| 1 | Master seed | `20261104` | The only source of randomness. The baseline's seed 0 was a placeholder and is unrelated. |
| 2 | Stream derivation | `numpy.random.SeedSequence(20261104, spawn_key=(role,))` | Role table below. New purposes get new role numbers; existing roles never change. |
| 3 | Input preprocessing | `x = (pixel / 255 − 0.1307) / 0.3081` | Input range is about [−0.4242, 2.8215]; every pixel is nonzero. **Still to confirm with Aneesh and Parnian** so both students' results compare. |
| 4 | Training recipe | Adam, lr 1e-3, batch 64, **10 epochs**, cross-entropy, no dropout, no weight decay, no LR schedule | The doc fixes only the architecture and 5–10 epochs. |
| 5 | FP16 margin | \|acc_fp16 − acc_fp32\| ≤ 0.1 percentage points (≤ 10 of 10,000 images) | Parnian may have her own threshold; Tony to ask. |
| 6 | Training device | CPU, FP32 | Chosen for repeatable training. FP16 inference and the campaign run on MPS. |
| 7 | Plot format | PNG, committed with `git add -f` | `.gitignore` excludes `*.png`. |

**Role table** (fixed; append only):

| Role | Purpose | Consumer |
|---|---|---|
| 0 | Training: weight init and per-epoch shuffling | torch CPU global RNG |
| 1 | Step 4: drawing the single weight | numpy `Generator` |
| 2 | Step 5: sampling 150 elements of `fc1.weight` | numpy `Generator` |
| 3 | Step 5: sampling 150 elements of `fc2.weight` | numpy `Generator` |

- numpy consumers: `np.random.default_rng(np.random.SeedSequence(MASTER, spawn_key=(role,)))`.
- torch consumers: `int(np.random.SeedSequence(MASTER, spawn_key=(role,)).generate_state(1)[0])` as the seed.
- No literal seeds anywhere else in the code. Record every derived seed in `run.json`.
- The doc lists "training" as one purpose, so init and shuffling share role 0. That matches the doc.

## 5. Interpretations where the doc is silent

These are not deviations, but state them in the README and the step 7 paragraph.

1. **Reference = fault-free FP16 predictions on MPS.** The doc says the reference is "the same trained network with no fault". The faulted network is FP16 on MPS, so the reference is the same FP16 network on MPS before any flip.
2. **MPS counts as the doc's "GPU runtime".**
3. **FP32 accuracy is measured twice:** on CPU right after training (step 1), and on MPS (step 2). The FP16 margin check compares FP16-on-MPS with FP32-on-MPS, so only precision differs, not device. Both numbers go in `run.json`.
4. **Step 4's "magnitude below 1" uses the FP16 value.** An FP32 weight such as 0.9999 rounds to 1.0 in FP16, and flipping bit 14 of 1.0 gives inf.
5. **`requirements-mac.txt`** holds the Mac stack. The doc names `requirements.txt`; that file stays as the PC stack.
6. **The evaluation device affects some classifications.** Whether a large exponent fault ends as `non_finite` or `sdc` can depend on how the backend accumulates the matrix product (FP16 or higher precision internally). Record the device and library versions; name the device in the step 7 paragraph.
7. **The step 3 sanity-check weight is chosen without randomness** (the `fc1.weight` element with the largest FP16 magnitude below 1), so it needs no RNG role. The doc only asks for "one weight that the network uses".

## 6. Machine and environment

Measured on Nov 7, 2026 (read-only checks):

| Item | Value |
|---|---|
| Machine | MacBook, Apple M5 Pro, `arm64` |
| macOS | 26.6.2 |
| Python for this work | `/opt/homebrew/bin/python3.12` (3.12.16). Python 3.12 matches the PC. |
| Pitfall | Plain `python3` is 3.14.5. Always create the venv with `python3.12`. |
| venv | Not created yet |

**Setup commands** (zsh, from the repo root):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib pytest
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())"
pip freeze > requirements-mac.txt
```

- The check must print `arm64 <version> True`. `x86_64` means an Intel Python under Rosetta: reinstall an arm64 Python.
- Reinstall later with `pip install -r requirements-mac.txt`. Put that line in the README.
- Libraries: PyTorch with MPS (same library as the diffusion work, so `src/bitflip.py` carries over), torchvision (MNIST download), numpy, matplotlib, pytest. Not MLX: the injector would need rewriting for diffusers next week.

## 7. Planned files

Simplify if it can be smaller.

| Path | Contents | Committed? |
|---|---|---|
| `src/bitflip.py` | `flip`, `restore`, `fingerprint`, bit-field names. Reused for the DDPM next week. | yes |
| `tests/test_bitflip.py` | Hand tests on CPU and MPS | yes |
| `src/mnist_fault.py` | Subcommands: `check-mps`, `train`, `one-weight`, `campaign`, `summarize` | yes |
| `configs/mnist.json` | All settings from section 4 | yes |
| `requirements-mac.txt` | `pip freeze` of the Mac venv | yes |
| `data/` | MNIST download (torchvision) | **no. Add `data/` to `.gitignore`**: the current `.gitignore` doesn't cover it |
| `logs/mnist/<run_name>/` | One folder per run (contents per step below) | CSVs, `run.json` and the plot yes; `*.pt`, `*.npy` no (already ignored) |

**Proposed `configs/mnist.json`:**

```json
{
  "master_seed": 20261104,
  "roles": {"train": 0, "one_weight": 1, "sample_fc1": 2, "sample_fc2": 3},
  "data": {"root": "data", "normalize_mean": 0.1307, "normalize_std": 0.3081},
  "model": {"layers": [784, 128, 10], "activation": "relu"},
  "train": {"device": "cpu", "dtype": "float32", "optimizer": "adam", "lr": 0.001,
            "batch_size": 64, "epochs": 10, "loss": "cross_entropy"},
  "eval": {"device": "mps", "dtype": "float16", "fp16_margin_pp": 0.1},
  "campaign": {"elements_per_matrix": 150, "tensors": ["fc1.weight", "fc2.weight"], "bits": 16}
}
```

**Proposed commands** (finalize once implemented; Tony runs them):

```bash
pytest tests/
python src/mnist_fault.py check-mps
python src/mnist_fault.py train --run-name train_a
python src/mnist_fault.py train --run-name train_b        # must hash identically to train_a
python src/mnist_fault.py one-weight --model logs/mnist/train_a/model_fp32.pt --run-name one_weight
python src/mnist_fault.py campaign --model logs/mnist/train_a/model_fp32.pt --run-name campaign_a
python src/mnist_fault.py campaign --model logs/mnist/train_a/model_fp32.pt --run-name campaign_b
python src/mnist_fault.py summarize logs/mnist/campaign_a
```

## 8. Steps

Step numbers follow the doc's cross-references (4 = one weight, 5 = campaign, 6 = summary, 7 = paragraph). The doc's own list numbering renders oddly; ignore it.

### Setup: Mac environment
**Status:** not started.
- Run the setup commands in section 6. Write `requirements-mac.txt`. Add `data/` to `.gitignore`.
- **Done when:** the check prints `arm64 ... True`, and `requirements-mac.txt` exists.

### Step 0: MPS capability check
**Status:** not started.
- Confirm on MPS:
  1. An FP16 matmul runs: `(torch.randn(64, 784, device="mps").half() @ torch.randn(784, 128, device="mps").half())` gives an FP16 result on `mps`.
  2. `w.view(torch.int16)` shares storage with the FP16 tensor `w`, both on MPS: writing a value through the int16 view changes `w` (compare before/after on CPU copies; also check `data_ptr()` is equal).
  3. The step 3 hand tests pass on MPS.
- **Stop if** any of these fail: tell Tony. The doc's fallback is CPU FP32 with 32 bits per element (9,600 faults, must be reported). CPU FP16 would keep 4,800 faults and the diffusion study's bit layout, but it needs Parnian's OK. Never switch quietly.

### Step 1: Train (CPU, FP32)
**Status:** not started.
- **Data.** `torchvision.datasets.MNIST(root="data", train=..., download=True)`. Preload `.data` (uint8, N×28×28) and `.targets` once; convert to `float32`, divide by 255, normalize with 0.1307 / 0.3081, reshape to N×784. Record the SHA-256 of the four **uncompressed** raw files in `data/MNIST/raw/` (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`). The uncompressed files are what torchvision and the doc's Kaggle copy (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) have in common.
- **Model.** `nn.Sequential`-style or a small module with `fc1 = nn.Linear(784, 128)`, ReLU, `fc2 = nn.Linear(128, 10)`. Parameter names must be `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`.
- **Assert** total parameters = 101,770 and bias parameters = 138.
- **Seeding (role 0).** `torch.manual_seed(seed_role0)` once, then build the model (init uses the global RNG), then draw each epoch's order with `torch.randperm(60000)` from the same global RNG. Index the preloaded tensors with it; no `DataLoader` workers. No other RNG use during training.
- **Train** 10 epochs: Adam (lr 1e-3), batch 64 (last batch smaller: 60,000 = 937 × 64 + 32), cross-entropy, no dropout.
- **Record** FP32 test accuracy on CPU (`fp32_cpu_accuracy`) on all 10,000 test images, and `torch.get_num_threads()`.
- **Save** `model_fp32.pt` (state dict; git-ignored). Record (a) the SHA-256 of the file and (b) a parameter hash: SHA-256 over the raw bytes of each tensor in `state_dict()` order. Use (b) for comparisons; it doesn't depend on file metadata.
- **Verify:** train twice (`train_a`, `train_b`) and confirm the parameter hashes are identical.
- **Stop if** the hashes differ: record it in `run.json` and the work log, and ask Tony.
- **Outputs:** `logs/mnist/train_a/{model_fp32.pt, run.json}`, same for `train_b`.

### Step 2: Switch the stored network to FP16 on MPS
**Status:** not started.
- Load `model_fp32.pt`. Make an FP32 copy on MPS and an FP16 copy on MPS (`.half().to("mps")`).
- Preload the 10,000 test images once as a `(10000, 784)` tensor on MPS, FP32 for the FP32 model and FP16 for the FP16 model. Each evaluation is one forward pass under `torch.no_grad()`.
- Record `fp32_mps_accuracy` and `fp16_mps_accuracy`. **Check** \|fp16 − fp32_mps\| ≤ 0.1 pp.
- **Reference:** the fault-free FP16 logits and predictions (`argmax(dim=1)`). Save the predictions (git-ignored `.npy` or `.pt`) and record their SHA-256 and the logits' SHA-256 in `run.json`.
- **Verify:** evaluate the clean FP16 network twice; the logits must be byte-identical (compare SHA-256 of `logits.cpu().numpy().tobytes()`).
- **Stop if** the margin fails or the logits differ: record and ask Tony.

### Step 3: Build the single-bit flip (`src/bitflip.py`)
**Status:** not started.
- **Address:** (tensor name, flat index in storage order, from 0). `nn.Linear` stores weights as (out, in):
  - `fc1.weight`: 128 × 784 = 100,352 elements; flat index `i` is hidden unit `i // 784`, pixel `i % 784`.
  - `fc2.weight`: 10 × 128 = 1,280 elements; flat index `i` is output class `i // 128`, hidden unit `i % 128`.
- **`flip(model, name, index, bit) -> int`:** under `torch.no_grad()`, get the live parameter (`dict(model.named_parameters())[name]`), take `p.data.view(-1).view(torch.int16)` (a view of the same device storage; never a copy), read the element's 16-bit pattern, XOR a single-bit mask, write it back into that view. Return the original pattern as a Python int.
- **`restore(model, name, index, pattern)`:** write the saved pattern back through the same int16 view. **Never restore by flipping again.**
- **`fingerprint(tensor) -> str`:** SHA-256 hex of the raw bytes (`tensor.detach().cpu().contiguous().numpy().tobytes()`). Copying to CPU to hash is fine; only the flip must hit the device tensor.
- **Field names:** bit 15 → `sign`, 14–10 → `exponent`, 9–0 → `mantissa`.
- **Pitfall:** the bit-15 mask `0x8000` = 32768 doesn't fit in int16. Use −32768. Writing 32768 raises `RuntimeError: value cannot be converted to type int16_t without overflow` (seen on CPU, torch 2.15).
- **Hand tests** (`tests/test_bitflip.py`, run on CPU and on MPS):

  | Start value | Bit | Expected | Source |
  |---|---|---|---|
  | 0.5 | 14 | 32768.0 | doc |
  | 1.0 | 14 | inf | doc |
  | 0.5 | 0 | 0.50048828125 | doc |
  | 0.5 | 15 | −0.5 | added (covers the bit-15 pitfall) |
  | any | any | flip then restore leaves the fingerprint unchanged | added |

- **Sanity check on the trained FP16 network (doc):** flip bit 14 of one `fc1.weight` element the network uses and confirm the outputs change; restore and confirm the logits and fingerprint match the reference. Choose the element without randomness so no RNG role is used: the `fc1.weight` element with the largest FP16 magnitude below 1. (With mean/std inputs every pixel is nonzero.) If the outputs never change, the code flipped a copy or the wrong tensor.
- **Stop if** a hand test or the sanity check fails.

### Step 4: Walk one weight through all 16 bits
**Status:** not started.
- With the role 1 generator, draw `rng.integers(0, 100352)` until the **FP16** value of that `fc1.weight` element has magnitude below 1. Record the chosen index and how many draws it took.
- For bits 0 to 15: fingerprint → flip → evaluate on all 10,000 test images → classify against the reference → restore → verify the fingerprint.
- **Output:** `logs/mnist/one_weight/one_weight.csv`, 16 rows, columns `bit, field, original_value, new_value, class, accuracy`, plus `run.json`. Use the same number formatting as step 5.
- **Stop if** any fingerprint check fails.

### Step 5: Run the campaign (4,800 faults)
**Status:** not started.
- **Choose the elements first.** `rng2.choice(100352, 150, replace=False)` for `fc1.weight` (role 2) and `rng3.choice(1280, 150, replace=False)` for `fc2.weight` (role 3). Keep draw order (don't sort). Write `chosen_elements.csv` with columns `tensor, element_index, original_value` (fc1 rows first, then fc2) **before any flipping**.
- **Loop:** for each of the 300 elements (CSV order), for bit 0..15:

  ```
  fp = fingerprint(tensor)
  try:
      flip; evaluate on all 10,000 test images; classify
  finally:
      restore
  if fingerprint(tensor) != fp: abort the whole run
  ```

  Any exception or failed check stops the whole run (no skipping, no retries).
- **`faults.csv`:** one row per fault, exactly these columns in this order: `tensor, element_index, bit, original_value, new_value, class, accuracy`. `new_value` is the FP16 value read back from the tensor after the flip.
- **Stable formatting:** values as `repr(float(v))` (gives `inf`, `-inf`, `nan` for non-finite); accuracy as `f"{correct / 10000:.4f}"`; class as `masked` / `sdc` / `non_finite`. Write with `newline=""` so line endings are the same in every run.
- **No run-specific data in the CSVs.** Timestamps, wall time and paths go in `run.json`, or the identical-file check fails for the wrong reason.
- **`run.json`:** command; git commit and dirty flag (same convention as `src/baseline.py`: `git status --porcelain --untracked-files=no`); master seed, role table and every derived seed; device; macOS version; chip (`sysctl -n machdep.cpu.brand_string`); Python, torch, torchvision and numpy versions; torch thread count; MNIST file hashes; model file and parameter hashes; reference prediction and logit hashes; `fp32_cpu_accuracy`, `fp32_mps_accuracy`, `fp16_mps_accuracy`; wall time.
- **Determinism (hand-in item):** run twice into `campaign_a` and `campaign_b`. `chosen_elements.csv` and `faults.csv` must be byte-identical: compare SHA-256 and save the comparison as `logs/mnist/campaign_compare.json`.
- **Expected cost:** 4,800 forward passes of 10,000 images through ~0.1M parameters (~2 GFLOP each). Should take minutes on MPS; measure and record.
- **Stop if** any check fails, or the two runs' CSVs differ (record it and ask Tony).

### Step 6: Summarize
**Status:** not started.
- Read only `faults.csv` (so the summary is reproducible from the committed file).
- Per bit (n = 300): count and share of each class, each with a 95% Wilson interval, z = 1.96:
  - centre = (p + z²/2n) / (1 + z²/n)
  - half-width = z·√(p(1 − p)/n + z²/4n²) / (1 + z²/n)
  - **Check value:** k = 0, n = 300 → [0, 1.26%]. Add this as a test.
- **Outputs:**
  - `per_bit.csv`: `bit, field, n`, then for each class `<class>_count, <class>_share, <class>_lo, <class>_hi`.
  - `per_field.csv`: same, grouped as sign (n = 300), exponent (n = 1,500), mantissa (n = 3,000). The doc asks for both tables.
  - `per_bit.png`: SDC share and non-finite share against bit position 0–15, with Wilson intervals as error bars and the sign / exponent / mantissa ranges marked. Commit with `git add -f`.
- **Caveat to note in the report:** the same 300 elements are reused for every bit, so faults inside a field group are not independent; the grouped intervals treat them as independent.

### Step 7: Fault-model paragraph
**Status:** not started. **Tony writes it.**
- It must let another student repeat the experiment from the paragraph alone. Per the doc it must cover: what was flipped, where, which bits, how many, how they were chosen, how randomness was controlled, how values were restored, and how faults were classified.
- **Agent's job:** check every fact against `configs/mnist.json`, `run.json` and this file (seed, role method, counts, device, preprocessing, interpretations in section 5). Report mismatches; don't rewrite the paragraph.

### Step 8: Document
**Status:** not started.
- Add a section for this week to `README.md`: master seed; machine (MacBook M5 Pro, macOS version, MPS); library versions (`requirements-mac.txt`, reinstall with `pip install -r requirements-mac.txt`); the exact commands that reproduce every table; where the result files are; the statement that two same-seed runs gave identical files (with the hashes); the interpretations in section 5; and that this is a separate result set from the PC.
- Update the README's status line and "no injector code exists yet" note.

## 9. Hand-in checklist (doc)

| Item | From | Status |
|---|---|---|
| Repo link; README with master seed, library versions and the reproduce command | Step 8 | not started |
| File of the 300 chosen elements | `chosen_elements.csv` (step 5) | not started |
| One-row-per-fault result file | `faults.csv` (step 5) | not started |
| Statement that two same-seed runs gave identical result files | Step 5 check | not started |
| One-weight table | Step 4 | not started |
| Per-bit table and plot | Step 6 | not started |
| Fault-model paragraph | Step 7 (Tony) | not started |
| Short answers: (1) which bits cause SDC and which cause non-finite outputs, explained from FP16 storage; (2) which bits never changed a prediction, and why; (3) how FP32 storage would change the results; (4) what in the fault model would need changing for a diffusion model | Tony, from the data | not started |

Also on the doc's list for Tony this week: reading on fault vs error vs failure, masked and unmasked faults, and silent data corruption.

## 10. Open items (Tony)

- [ ] Confirm input preprocessing (mean/std) with Aneesh and Parnian. Blocks step 1 only.
- [ ] Ask Parnian whether the 0.1 pp FP16 margin is acceptable. Needed at step 2.
- [ ] Add Aneesh and Parnian as GitHub collaborators (a doc task item).
- [ ] Decide whether the baseline PNGs force-added in `c611a20` stay in the repo.

## 11. Decided against (and why)

- **FP16 on the CPU instead of MPS:** the doc ties CPU use to FP32 storage; the bit layout is identical on any device; repeatability on MPS is already tested by steps 2 and 5; neither device matches next week's NVIDIA GPU. Reconsider only if step 0 or a determinism check fails, and only with Parnian's OK.
- **Training on MPS:** no documented reproducibility guarantee found for MPS training, and no speed benefit for a network this small.
- **A second campaign in FP32 (9,600 faults):** not asked for; short answer 3 asks for reasoning, not a measurement. Possible later experiment if Parnian wants data.
- **Adding "pixel is nonzero" to the step-4 draw:** would change a doc rule; unnecessary with mean/std inputs, where every pixel is nonzero.

## 12. Known pitfalls

- Bit-15 mask must be −32768 in int16 (step 3).
- `python3` is 3.14 on this Mac; use `python3.12`.
- `data/` isn't git-ignored yet; add it before downloading MNIST.
- `*.png` is git-ignored; the plot needs `git add -f`.
- MPS ops are asynchronous; `.item()` and `.cpu()` synchronize. Read results only through those.
- `p.data.view(-1)` requires a contiguous tensor; `nn.Linear` weights are contiguous, but assert it.
- Keep timestamps, paths and wall time out of the CSVs (step 5).

## 13. Next week (context only)

Nov 9 – 16 runs on the Windows RTX 5090 PC: the DDPM (`google/ddpm-cifar10-32`) fault injector, dtype audit, and a first demo of bit 14 and bit 0 flips in 4 UNet regions. The DDPM `flip_bit` reuses `src/bitflip.py`; the FP16 bit layout is the same on NVIDIA.

## 14. Work log

Append one row per step or significant action. Newest last.

| Date | Who | Action | Result | Commit |
|---|---|---|---|---|
| 2026-11-07 | Claude (Mac) | Cloned repo; read `handoff-mnist-week1.md`, README, `src/`, `logs/` | Repo at `2484ce4`. Checked the handoff's numbers (hand tests, parameter counts, Wilson check value, \|v\| < 1 rule): all correct. | `2484ce4` |
| 2026-11-07 | Claude (Mac) | Read-only environment check | arm64, Apple M5 Pro, macOS 26.6.2, `python3.12` 3.12.16 present, no venv | `2484ce4` |
| 2026-11-07 | Claude (Mac) | Read the Google Doc Proposal tab; compared with the handoff | No conflicts. Doc leaves the 7 decisions open. Training is one purpose (one stream). Kaggle MNIST link confirmed. No comments or suggested edits returned. | `2484ce4` |
| 2026-11-08 | Tony + Claude | Settled the 7 decisions (section 4) | Seed 20261104; SeedSequence roles; mean/std; Adam 10 epochs; 0.1 pp; CPU training; PNG. | `2484ce4` |
| 2026-11-08 | Tony + Claude | Discussed CPU FP16 and FP32-on-GPU | Keep FP16 on MPS. Add FP32-on-MPS accuracy for the margin check (section 5, item 3). | `2484ce4` |
| 2026-11-08 | Claude (Mac) | Wrote this plan and `README-mnist-week1.md` | Committed with the pointer files (next row). | see next row |
| 2026-11-08 | Claude (Mac) | Added one-line pointers to this file in `AGENTS.md` and `CLAUDE.md`; committed and pushed all four files at Tony's request | `AGENTS.md` is read by most AI coding tools, `CLAUDE.md` by Claude Code. This Mac has no git identity configured, so the commit used Tony's identity from earlier commits via `git -c` (no config changed). | the commit that added this row |
