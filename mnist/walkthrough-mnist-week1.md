# Walkthrough: the MNIST bit-flip study (week of Nov 4 – 9, 2026)

A plain-language summary of what was done this week and why, written for a reader who knows basic neural networks and not much else. Written Nov 8, 2026; the results come from the official runs at commit `757c67a`. For full detail see [`plan-mnist-week1.md`](plan-mnist-week1.md); for the results tables and the plot see [`README.md`](README.md).

## The big picture

A neural network's weights sit in computer memory as bits (0s and 1s). A hardware glitch, or an attack, can flip one of those bits, so one weight silently changes value. The team's research question is what that does to diffusion models (image generators). This week was a practice run on something much smaller: a simple network that recognizes handwritten digits (MNIST). The point was to build and test the fault-injection tool cheaply before using it on the big models.

A flipped bit can lead to one of three outcomes:
- **Masked:** nothing visible happens; every answer stays the same.
- **Non-finite:** the network outputs infinity or "not a number". It's broken, but obviously so, and easy to detect.
- **Silent data corruption (SDC):** the outputs look normal, but some answers changed. Nothing warns you, which makes this the dangerous case.

## A few ideas used throughout

- **FP16 (16-bit floating point).** Each weight is stored in 16 bits, a bit like scientific notation (−1.23 × 10⁴) but in base 2:
  - bit 15 is the **sign** (+ or −),
  - bits 14–10 are the **exponent** (the scale, roughly "which power of 2"),
  - bits 9–0 are the **mantissa** (the precise digits).
- **Seed.** Computers make "random" choices from a starting number called a seed. Same seed, same choices. We used a single master seed, `20261104`, for everything, with a separate stream for each purpose (training, picking weights, and so on). Anyone can then reproduce the exact same results.
- **Fingerprint (hash).** A short code computed from a file or a block of memory. If even one bit changes, the code changes completely. We used it to prove a weight was put back exactly, and that two result files are identical.
- **CPU vs GPU.** The Mac's main processor vs its graphics chip. On a Mac, PyTorch reaches the GPU through a backend called MPS. The team doc says to run the 16-bit tests on a GPU.
- **Confidence interval.** If 60 of 300 tests show something, the true rate isn't exactly 20%. The 95% interval gives the range it plausibly lies in. We used the Wilson method the doc asked for.

## What was done, step by step

**1. Read the instructions and set the ground rules.** Claude cloned the repo and read the handoff note and the team Google Doc. They agreed on everything, but the doc left 7 choices open. Tony made them:
- the seed (`20261104`)
- how the random streams are derived from it
- input scaling (standard MNIST normalization)
- the training recipe (Adam, 10 passes over the data)
- the allowed FP16 accuracy drop (0.1 percentage points)
- training on the CPU
- PNG for the plot

**2. Wrote the documentation.**
- A detailed plan for AI tools to follow ([`plan-mnist-week1.md`](plan-mnist-week1.md)). It includes a work log updated after every step.
- A short guide for people ([`README.md`](README.md)).
- Pointer files (`AGENTS.md`, `CLAUDE.md` at the repo root) so other AI tools find the plan automatically.

**3. Set up the Mac.** Python 3.12 and PyTorch 2.15 (the same PyTorch version as the PC) were installed, and the Mac's GPU was confirmed to work. The exact library versions are recorded in [`requirements-mac.txt`](requirements-mac.txt), so anyone can reinstall the same setup.

**4. Built and tested the bit-flip tool ([`src/bitflip.py`](src/bitflip.py)).** It has three functions:
- **flip:** inverts one chosen bit of one chosen weight, directly in the memory the network uses (not a copy).
- **restore:** puts the original bits back.
- **fingerprint:** takes a hash, so we can prove the restore was exact.

It was tested on examples you can check by hand. Flipping bit 14 of 0.5 must give 32,768. Flipping bit 14 of 1.0 must give infinity. All 47 tests passed on both the CPU and the GPU.

**5. Repo reorganized.** A second Claude session split the repo into `cifar10/` (the diffusion work) and `mnist/` (this study). The two sessions coordinated so neither broke the other's files.

**6. Trained the network** (784 inputs → 128 hidden units → 10 outputs) on the CPU. It took about 4 seconds. It was trained twice: both runs gave identical weights, which proves training is repeatable. Test accuracy was **97.43%**.

**7. Switched to 16-bit on the GPU.** Accuracy stayed at **97.43%**, so storing weights in FP16 cost nothing. The network's answers on all 10,000 test images were saved as the **reference**: every fault is judged against them.

Then a sanity check: flip one important bit, confirm the answers change, restore it, and confirm everything matches the reference exactly. If the answers hadn't changed, the tool would have been flipping a copy instead of the real weight.

**8. One weight through all 16 bits.** One weight was picked with a seeded random draw. Then, for each of its 16 bits: flip it, test all 10,000 images, record what happened, and put it back. Only bit 14 changed any answer. That's the 16-row table in [`logs/one_weight/one_weight.csv`](logs/one_weight/one_weight.csv).

**9. The main experiment: 4,800 faults.**
- 300 weights were picked at random: 150 from each of the two weight layers.
- Each of their 16 bits was flipped one at a time, giving 300 × 16 = 4,800 tests.
- Each test runs all 10,000 images, sorts the result into masked, SDC or non-finite, restores the weight, and checks its fingerprint.
- The whole thing took about 11 seconds on the GPU.
- It was run twice with the same seed, and both runs produced **byte-for-byte identical** result files. That's a hand-in item.

**10. Summary and plot.** The outcomes were counted for each bit position, with confidence intervals, and grouped as sign, exponent and mantissa. The SDC and non-finite rates were plotted against bit position.

**11. Checked independently.** A separate script that doesn't use the main code re-picked the 300 weights from the seed, recomputed all 4,800 flipped values, recounted the table, and re-ran some faults from scratch. Everything matched.

**12. Practice runs, then official runs.** Everything was first run as practice. Then the code was saved to git, and all the runs were redone on that exact saved version. That way every result file records a commit that contains the code that made it.

## Results in plain words

- **Overall:** of 4,800 single-bit faults, 3,437 were masked (72%), 1,264 caused silent data corruption (26%), and 99 gave non-finite outputs (2%).
- **By field:**

  | Field | Tests | Masked | SDC | Non-finite |
  |---|---|---|---|---|
  | Sign (bit 15) | 300 | 125 | 175 | 0 |
  | Exponent (bits 14–10) | 1,500 | 587 | 814 | 99 |
  | Mantissa (bits 9–0) | 3,000 | 2,725 | 275 | 0 |

- **What the table shows:**
  - Bits 0–2 never changed a single answer.
  - From bit 3 upward, the SDC rate mostly rises, reaching 76% at bit 12.
  - All 99 non-finite cases came from bit 14.

The full per-bit table and plot are in the [README](README.md). Explaining *why* the bits behave this way is the subject of Tony's four short answers, so this report deliberately leaves that reasoning out.

## Where everything is

- [`README.md`](README.md): plain summary, how to rerun everything, and the results with tables and the plot.
- [`plan-mnist-week1.md`](plan-mnist-week1.md): the detailed plan, the choices made, and the work log of every step.
- [`logs/campaign_a/`](logs/campaign_a/): `chosen_elements.csv` (the 300 weights), `faults.csv` (one row per fault), `per_bit.csv`, `per_field.csv`, `per_bit.png`.
- [`logs/one_weight/one_weight.csv`](logs/one_weight/one_weight.csv): the 16-row single-weight table.

## Saved to GitHub

| Commit | What it holds | Committed by |
|---|---|---|
| `72e467b` | Plan, plain README, AI pointer files | Claude |
| `f851270` | Repo split into `cifar10/` and `mnist/` | second Claude session, with Tony's OK |
| `f1f6bb5` | Mac setup and the bit-flip tool | second Claude session, with Tony's OK |
| `757c67a` | Training, experiment and summary code | Claude |
| `0ea6d5a` | Official results and the README results section | Claude |

This report was added in a later commit.

## What's left

1. **The fault-model paragraph (step 7):** in Tony's own words, describe what was flipped, where, which bits, how many, how they were chosen, how randomness was controlled, how weights were restored, and how outcomes were classified. Another student should be able to repeat the experiment from it alone. An AI agent then checks every fact against the run records.
2. **The four short answers** (Tony):
   - which bits cause SDC and which cause non-finite outputs, and why, from the way FP16 stores numbers
   - which bits never changed a prediction, and why
   - how 32-bit storage would change the results
   - what in the fault model would need changing for a diffusion model
3. **Confirm the input scaling** with Aneesh and Parnian, and the 0.1-point FP16 limit with Parnian. If they want something different, rerunning everything takes under a minute.
4. **Add Aneesh and Parnian** as collaborators on the GitHub repo.
