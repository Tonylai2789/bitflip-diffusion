# Handoff: MNIST bit-flip injector week (Nov 4 – Nov 9, 2026) on the MacBook M5 Pro

**For:** a Claude Code session (or Tony) on Tony's MacBook (Apple M5 Pro), in Tony's repo `Tonylai2789/bitflip-diffusion`.
**From:** the dev-box session, Nov 6, 2026.
**Source of truth:** the "Proposal" tab of the team Google Doc *Resilience of Diffusion Models Under Bit Flip Faults*, section **Tasks (Nov 4 – Nov 9)**: https://docs.google.com/document/d/1q3cozz_Vtl_ZVSSwx1kGn-ljIXv3hj5XidglNhFjE9E/edit?tab=t.0. Where this file and the doc disagree, the doc wins. Do not edit the doc.
**Due:** Monday, Nov 9.

Step numbers below follow the doc's own cross-references (step 4 = one weight, 5 = campaign, 6 = summary, 7 = paragraph). The doc's list numbering renders oddly (1, 1, 1, 1, 2, 3, ...), so ignore it.

## 1. Where things stand

- Repo `https://github.com/Tonylai2789/bitflip-diffusion` (private), at `d183864` or later. It has the README with the research questions, `.gitignore` (model weights, `*.png`, `*.jpg`, `*.npy`), `src/`, `configs/`, `logs/`, `notebooks/`.
- `requirements.txt` is pinned for the **Windows RTX 5090 PC** (`torch==2.15.0+cu134`). It will not install on a Mac. Leave it alone; the Mac gets its own file (section 4).
- Already done, ahead of schedule: the DDPM clean baseline (now a Nov 9–16 task). 64 DDIM images on the PC, two runs bitwise identical (`logs/seed0_a`, `logs/seed0_b`).
- New this week: Parnian's fault model on a small MNIST network. No code exists for it yet.
- The diffusion work (Nov 9–16) stays on the PC. This week's results come from the Mac. That makes them a separate result set with its own machine and pinned stack. Record that in the README. Don't mix Mac and PC runs inside one set.

## 2. Working rules

- **Tony commits and pushes.** Prepare files, run the checks and give exact commands. Don't commit, push or create repos unless Tony asks in this session.
- **Follow the fault model exactly.** The doc says: "Use the rules below exactly. Do not change them without writing down the reason." Any deviation needs a written reason in the README and the run record.
- **Ask before settling anything in section 3.**
- **Tony writes the step 7 paragraph and the short answers himself.** The doc says "in your own words." The session supplies the numbers and checks the facts.
- Record the git commit hash with every result.
- Tony's `~/.claude/CLAUDE.md` (Karpathy guidelines) applies: minimal code, surgical changes, verify each step.

## 3. Decisions for Tony before any code (suggested default in brackets)

1. **Master seed.** One integer, written in the README. It is the only source of randomness in the study. [Tony's choice. The baseline's seed 0 was a placeholder, not this.]
2. **How streams are derived from it.** The doc requires one stream per purpose, never shared. Adding a purpose later must not move existing streams. The doc mentions a "role number" but not the method. [numpy `SeedSequence(master_seed, spawn_key=(role,))` with a fixed role table: 0 = training, 1 = step-4 single weight, 2 = sampling `fc1.weight`, 3 = sampling `fc2.weight`. New purposes get new numbers. A torch generator takes its seed from `SeedSequence(...).generate_state(1)[0]`. No literal seeds anywhere else.]
3. **Input preprocessing.** This matters more than it looks, so agree it with Aneesh and Parnian so the two students' results compare. [Pick one and record it.]
   - Pixel scaling multiplies every first-layer product, so it moves high-exponent faults between "non-finite" and "silent data corruption".
   - With `[0, 1]` scaling (pixel / 255), border pixels are 0 in every image. A finite flipped weight on such a pixel can then never change anything.
   - With standard normalization (mean 0.1307, std 0.3081), every pixel is nonzero.
4. **Training recipe.** The doc fixes only the architecture and 5–10 epochs. [Adam, lr 1e-3, batch 64, 5 epochs, cross-entropy, no dropout. Keep all of it in a config file.]
5. **FP16 "small margin".** The doc gives no number. [|acc_fp16 − acc_fp32| ≤ 0.1 percentage points. Ask Parnian if she has a threshold.]
6. **Training device.** [CPU, in FP32. The network is tiny, so CPU training takes seconds per epoch, and I could find no documented reproducibility guarantee for MPS training. FP16 inference and the campaign run on the Mac GPU (MPS), as the doc requires.]
7. **Plot format.** `.gitignore` excludes `*.png`, but the hand-in needs the plot. [Save it as `.svg` or `.pdf` so it commits normally.]

Still open from earlier (Tony only):
- Add Aneesh and Parnian as collaborators on the GitHub repo.
- Decide whether the baseline PNGs force-added in `c611a20` stay in the repo.

## 4. Mac setup

**Libraries.** All are open source, run locally and need no CUDA:
- **PyTorch with the MPS backend** (Apple's Metal GPU). Supports FP16 tensors on the Apple GPU. It is the same library as the diffusion work, so `src/bitflip.py` carries over to the DDPM next week.
- **torchvision** downloads MNIST. The doc links the Kaggle copy of the same original files; either source is fine, but record the SHA-256 of the four raw files.
- **numpy**, **matplotlib**, and **pytest** for the tests.
- Not recommended: **MLX** (Apple's own framework). It also does FP16 on the GPU, but the injector would have to be rewritten in PyTorch for diffusers next week.

**Commands** (zsh). Cloning the private repo needs GitHub auth on the Mac (`gh auth login` or a credential helper).

```bash
git clone https://github.com/Tonylai2789/bitflip-diffusion.git
cd bitflip-diffusion
python3.12 -m venv .venv          # if missing: brew install python@3.12
source .venv/bin/activate
pip install torch torchvision numpy matplotlib pytest
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())"
pip freeze > requirements-mac.txt
```

- The check must print `arm64 ... True`. If the machine type is `x86_64`, the Python is an Intel build running under Rosetta; reinstall an arm64 Python.
- Python 3.12 matches the PC. The doc doesn't require it.
- Add the reinstall line to the README: `pip install -r requirements-mac.txt`.

## 5. Plan

**Suggested files** (simplify if it can be smaller):
- `src/bitflip.py`: flip, restore, fingerprint. Reused for the DDPM next week.
- `tests/test_bitflip.py`: the hand tests.
- `src/mnist_fault.py`: train, one-weight walk, campaign, summarize (subcommands, or separate scripts).
- `configs/mnist.json`: master seed, role table, preprocessing, training recipe, 150 elements per matrix.
- Outputs under `logs/mnist/<run_name>/`.

### Step 0: Capability check on MPS (before anything else)

Confirm:
- FP16 matmul runs on MPS.
- `w.view(torch.int16)` on an MPS FP16 tensor shares storage: writing through the int16 view changes the FP16 tensor, with both on MPS.
- The step 3 hand tests pass on MPS.

If any of this fails, **stop and tell Tony.** The doc's fallback is the CPU with **FP32 storage and 32 bits per element** (bit 31 sign, 30–23 exponent, 22–0 mantissa). That gives 9,600 faults, and the report must say so. Don't quietly switch to CPU FP16.

### Step 1: Train

- Network: 784 → 128 → ReLU → 10 (two `nn.Linear`). The training stream (role 0) seeds both weight init and data shuffling.
- In code, assert the parameter count is 101,770 and the bias count is 138.
- Record the FP32 accuracy on all 10,000 test images.
- Save the parameters to `model_fp32.pt` (git-ignored; record its SHA-256 in `run.json`).
- **Verify:** train twice and confirm identical parameter hashes. If they differ, record it and ask.

### Step 2: Switch the stored network to FP16

- Use `.half()` and move to MPS. Inputs are FP16 on MPS too.
- Preload the 10,000 test images once as a `(10000, 784)` tensor, so each evaluation is a single forward pass.
- FP16 accuracy must be within the margin (decision 5) of FP32.
- The fault-free FP16 predictions are the reference for every classification. Save them.
- **Verify:** evaluate the clean network twice and confirm the logits are byte-identical.

### Step 3: Build the single-bit flip

- **Address:** (tensor name, flat index in storage order, from 0). `nn.Linear` stores weights as (out, in):
  - `fc1.weight` is 128 × 784 = 100,352 elements; flat index `i` is hidden unit `i // 784`, pixel `i % 784`.
  - `fc2.weight` is 10 × 128 = 1,280 elements.
- **`flip(model, name, index, bit)`:**
  - Reads the element's 16-bit pattern through an int16 view of the live parameter tensor, XORs a single-bit mask and writes it back into the same tensor inference uses. Never a copy.
  - Returns the original pattern.
- **`restore(model, name, index, pattern)`:** writes the saved pattern back. Never by flipping again.
- **`fingerprint(tensor)`:** SHA-256 of the raw bytes. Copying to CPU to hash is fine; only the flip must hit the device tensor.
- **Pitfall:** the bit 15 mask is `0x8000` = 32768, which doesn't fit in a signed int16. Use its signed equivalent (−32768). Writing 32768 directly raises `RuntimeError: value cannot be converted to type int16_t without overflow` (checked on CPU, torch 2.15).
- **Hand tests:**
  - From the doc: 0.5 with bit 14 → 32768.0; 1.0 with bit 14 → inf; 0.5 with bit 0 → 0.50048828125.
  - Added here: 0.5 with bit 15 → −0.5 (covers the pitfall); flip then restore → fingerprint unchanged.
- **Sanity check (doc):**
  1. Flip bit 14 of one `fc1` weight whose pixel is nonzero in the test images, and confirm the outputs change.
  2. Restore, and confirm the outputs and fingerprint match the reference.
  3. If the outputs never changed, the code flipped a copy or the wrong tensor.

### Step 4: Walk one weight through all 16 bits

- Using the role 1 stream, draw flat indices from `fc1.weight` until the FP16 value's magnitude is below 1. This keeps the bit 14 flip finite: the largest result is 65504, while values in [1, 2) would become inf or NaN.
- For bits 0 to 15: fingerprint, flip, evaluate, classify, restore, verify the fingerprint.
- Output a 16-row table with columns: bit, field (sign / exponent / mantissa), original value, new value, class, accuracy.

### Step 5: Run the campaign

- **Choose the elements.** Draw 150 distinct flat indices from `fc1.weight` (role 2) and 150 from `fc2.weight` (role 3), uniform and without replacement (`rng.choice(n, 150, replace=False)`). Write them to `chosen_elements.csv` (tensor, element_index, original_value) before any flipping.
- **Run all 300 elements × bits 0–15 = 4,800 faults:**

  ```
  fp = fingerprint(tensor)
  try:
      flip, evaluate on all 10,000 test images, classify
  finally:
      restore
  if fingerprint(tensor) != fp: abort the whole run
  ```

  Any exception or failed check also stops the whole run.
- **Classification**, in this order:
  1. Any logit is inf or NaN → `non_finite`.
  2. Otherwise, any prediction differs from the reference → `sdc`.
  3. Otherwise → `masked`.
- **`faults.csv`:** one row per fault, exactly these columns: tensor, element_index, bit, original_value, new_value, class, accuracy.
- **Determinism (a hand-in item):** run the whole campaign twice with the same seed into separate folders. `chosen_elements.csv` and `faults.csv` must be byte-identical; compare their SHA-256.
- **Keep run-specific data out of the CSVs.** Timestamps, wall time and paths go in `run.json`, or the identical-file check fails for the wrong reason. `run.json` holds:
  - Commit hash and dirty flag.
  - Master seed and the derived seeds.
  - Device, macOS version, chip (`sysctl -n machdep.cpu.brand_string`), Python and library versions.
  - MNIST file hashes, model hashes, FP32 and FP16 accuracy, wall time.
- **Use stable number formatting.** For example, `repr(float(v))` for values, and accuracy as `correct / 10000` to 4 decimals.

### Step 6: Summarize

- For each bit (300 faults each): the count and share of each class, with a 95% Wilson interval on every share.
- **Wilson interval**, with z = 1.96:
  - centre = (p + z²/2n) / (1 + z²/n)
  - half-width = z·√(p(1 − p)/n + z²/4n²) / (1 + z²/n)
  - Check value: k = 0, n = 300 gives [0, 1.26%].
- **Outputs:**
  - `per_bit.csv`, plus the same summary grouped as sign / exponent / mantissa (the doc asks for both).
  - A plot of the SDC share and the non-finite share against bit position (0–15), with the intervals and the sign / exponent / mantissa ranges marked.

### Step 7: Fault model paragraph

Tony writes it. It must let another student repeat the experiment from the paragraph alone. The session checks every fact in it against `run.json` and the config.

### Step 8: Document

Add a README section for this week:
- Master seed and machine (MacBook M5 Pro, macOS version, MPS).
- Library versions (`requirements-mac.txt`).
- The exact commands that reproduce every table.
- Where the result files are.
- The two-run identical statement.

## 6. Hand-in checklist (from the doc)

| Item | Comes from |
|---|---|
| Repo link; README with master seed, library versions and the reproduce command | Step 8 |
| File of the 300 chosen elements | `chosen_elements.csv` (step 5) |
| One-row-per-fault result file | `faults.csv` (step 5) |
| Statement that two same-seed runs gave identical result files | Step 5 check |
| One-weight table | Step 4 |
| Per-bit table and plot | Step 6 |
| Fault model paragraph | Step 7 (Tony) |
| Short answers: (1) which bits cause SDC and which cause non-finite outputs, explained from FP16 storage; (2) which bits never changed a prediction, and why; (3) how FP32 storage would change the results; (4) what in the fault model would need changing for a diffusion model | Tony, from the data |

Also on the doc's list for this week, as reading for Tony: fault vs error vs failure; masked and unmasked faults; silent data corruption.

## 7. Out of scope this week (doc)

- Faults in activations, inputs, biases or arithmetic hardware.
- More than one bit flipped at once.
- Any diffusion work on the Mac.

## 8. Notes

- **Expected cost:** 4,800 forward passes of 10,000 images through a 0.1M-parameter network. That should take minutes on MPS; measure it and record it in `run.json`.
- **Next week (Nov 9–16)** is on the PC. The DDPM `flip_bit` reuses `src/bitflip.py`, and the FP16 bit layout is the same on NVIDIA.
