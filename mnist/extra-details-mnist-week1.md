# Extra details (not the results file)

The results, the fault-model paragraph and the four short answers are in [`results-2026-11-09.md`](results-2026-11-09.md). This file keeps supporting details that were taken out of it to keep it short. Everything here comes from the same official runs at commit `757c67a`.

## Network size

784 → 128 (ReLU) → 10: 101,770 parameters, 138 of them biases.

## Random streams

Master seed `20261104`. Each purpose gets its own stream, derived with numpy `SeedSequence(master_seed, spawn_key=(role,))`:

| Role | Purpose | Derived seed |
|---|---|---|
| 0 | Training | 3501612813 |
| 1 | Choosing the step 4 weight | 1608880895 |
| 2 | Sampling `fc1.weight` | 2531023520 |
| 3 | Sampling `fc2.weight` | 4077134792 |

Each run folder's `run.json` records the commit.

## Checks on the fault-free network

- Two evaluations of the fault-free FP16 network gave identical outputs.
- **Sanity check:** flipping bit 14 of `fc1.weight[89696]` (-0.9482421875 → -62,144.0) changed 5,165 predictions (accuracy 46.85%). After the restore, the outputs and the fingerprint matched the fault-free network again.
- Training twice gave identical weights: parameter SHA-256 `4ee3056c93ccf1a2a56df7879c3f79464253091babe1bf913ff287c5531db923` ([`logs/train_compare.json`](logs/train_compare.json)).

## How the step 4 weight was picked

It was drawn from `fc1.weight` with its own seeded stream; the first draw already had magnitude below 1. Each bit was flipped, evaluated and restored before the next.

## Results by field

From [`logs/campaign_a/per_field.csv`](logs/campaign_a/per_field.csv). Each cell is the count, then the share with its 95% Wilson confidence interval.

| Field | Faults | Masked | SDC | Non-finite |
|---|---|---|---|---|
| sign | 300 | 125 (41.7% [36.2, 47.3]) | 175 (58.3% [52.7, 63.8]) | 0 (0.0% [0.0, 1.3]) |
| exponent | 1,500 | 587 (39.1% [36.7, 41.6]) | 814 (54.3% [51.7, 56.8]) | 99 (6.6% [5.5, 8.0]) |
| mantissa | 3,000 | 2,725 (90.8% [89.7, 91.8]) | 275 (9.2% [8.2, 10.3]) | 0 (0.0% [0.0, 0.1]) |

The same 300 elements are reused for every bit, so faults within a field aren't independent; the field intervals treat them as if they were.
