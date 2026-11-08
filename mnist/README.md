# MNIST bit-flip study (week of Nov 4 – Nov 9, 2026)

A short guide for people. The full step-by-step plan, written for AI agents, is in [`plan-mnist-week1.md`](plan-mnist-week1.md).

**Status (Nov 8):** decisions made, Mac setup done, bit-flip code started. The plan's work log has the details. **Due:** Monday, Nov 9.

## What we're doing

Computers store a neural network's weights as bits in memory. A hardware fault can flip one of those bits, changing a weight. We want to know which bits matter: does a flip break the network visibly, change its answers without warning, or do nothing?

This week we test that on a small digit-recognition network before trying it on diffusion models next week.

1. Train a small network to recognize handwritten digits (MNIST).
2. Store its weights as 16-bit floating point numbers (FP16), the format diffusion models usually use.
3. Pick 300 weights at random (150 from each layer). For each one, flip each of its 16 bits, one at a time, and check how the network's answers on 10,000 test images change. That's 4,800 tests.
4. Put every test into one of three groups:
   - **Non-finite:** the network outputs infinity or "not a number". Broken, but easy to detect.
   - **Silent data corruption:** the outputs look normal, but at least one answer changed. The dangerous case.
   - **Masked:** no answer changed.
5. Count the groups for each bit position and plot the results.

After every test the original bit is put back and checked, so each test starts from the same unchanged network.

## How an FP16 number is stored

| Bits | Part | What it does |
|---|---|---|
| 15 | Sign | Positive or negative |
| 14 – 10 | Exponent | Sets the size (roughly, the power of 2) |
| 9 – 0 | Mantissa | Sets the precise digits |

Bit 0 is the rightmost bit.

## Choices we made

| Choice | What we picked |
|---|---|
| Master seed (the one number all randomness comes from) | `20261104` |
| Network | 784 inputs → 128 hidden units → 10 outputs |
| Training | Adam optimizer, 10 passes over the training data, on the Mac's CPU |
| Input scaling | Standard MNIST normalization (to confirm with Aneesh and Parnian) |
| Where the faults are tested | The Mac's GPU, in FP16 |
| Allowed accuracy drop from switching to FP16 | At most 0.1 percentage points (to confirm with Parnian) |
| Plot format | PNG |

## Why the seed matters

Every random choice (training, which weights get picked) comes from one master seed, through a separate stream for each purpose. Anyone with the seed, the code and the same library versions gets exactly the same network, the same 300 weights and the same results. To prove it, we run the whole test twice and check the result files are identical.

## The machine

All of this week's results come from Tony's MacBook (Apple M5 Pro, macOS 26.6.2). They're a separate set from the diffusion results, which come from the Windows PC with an RTX 5090.

## What gets handed in

- This repository, with the master seed, library versions and the commands to reproduce everything.
- The list of the 300 chosen weights.
- The results file: one row for each of the 4,800 tests.
- Confirmation that two runs gave identical files.
- A 16-row table for a single weight, going through all its bits.
- The per-bit table and plot.
- A paragraph describing the fault model, and four short answers (written by Tony).

## How to run it

*Coming once the code exists.* The commands will go here and in the main [`README.md`](../README.md).

## Who does what

- **Tony:** commits and pushes; writes the fault-model paragraph and short answers; confirms the open choices with Aneesh and Parnian.
- **Claude (AI assistant):** writes the code, runs the checks, supplies the numbers and keeps [`plan-mnist-week1.md`](plan-mnist-week1.md) up to date.

## Sources

- Team Google Doc, "Proposal" tab, section *Tasks (Nov 4 – Nov 9)*. It has the final say on the rules.
- [`handoff-mnist-week1.md`](handoff-mnist-week1.md): the earlier handoff note.
