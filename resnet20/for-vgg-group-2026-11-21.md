# ResNet-20 bit-flip results, for the VGG group to compare (draft, Nov 21, 2026)

**Draft for review.** This file has our half of the Week 2 grid (ResNet-20 on Fashion-MNIST and CIFAR-10), exactly how we got it, and what we need from you so the two rows can be compared. Please read sections 2 and 5, then fill in the table in section 6.

| | Fashion-MNIST | CIFAR-10 |
|---|---|---|
| **ResNet-20** | ours (this file) | ours (this file) |
| **Small VGG-style network** | yours | yours |

The full write-up with figures is [`results-2026-11-20.md`](results-2026-11-20.md). All numbers here come from the official runs at commit `27ee894` (clean tree). The logs are in [`logs/`](logs/) (`fmnist_*` and `cifar10_*`).

## 1. Our results in one table

Each dataset: 9,600 faults (600 weights × 16 bits), each one evaluated on all 10,000 test images.

| | Fashion-MNIST | CIFAR-10 |
|---|---|---|
| FP32 test accuracy | 93.33% | 91.38% |
| FP16 test accuracy (the fault-free reference) | 93.32% | 91.35% |
| Masked | 5,017 (52.3%) | 3,300 (34.4%) |
| SDC | 4,525 (47.1%) | 6,172 (64.3%) |
| Non-finite | 58 (0.6%) | 128 (1.3%) |
| SDC where accuracy fell ≤ 0.1 pp (or rose) | 4,193 | 5,700 |
| SDC where accuracy fell > 0.1 pp | 332 | 472 |
| "Damaging" (SDC with a drop > 0.1 pp, or non-finite) | 390 (4.1%) | 600 (6.3%) |
| Share of damaging faults caused by bit 14 | 93% | 79% |
| Identical result files on a second run | yes | yes |

**SDC by FP16 field** (share of faults in that field, with 95% Wilson interval):

| Field (bits) | Fashion-MNIST SDC | Fashion-MNIST non-finite | CIFAR-10 SDC | CIFAR-10 non-finite |
|---|---|---|---|---|
| Sign (15), 600 faults | 77.3% [73.8, 80.5] | 0 | 95.2% [93.1, 96.6] | 0 |
| Exponent (14–10), 3,000 faults | 75.5% [74.0, 77.0] | 58 | 89.9% [88.8, 90.9] | 128 |
| Mantissa (9–0), 6,000 faults | 29.9% [28.8, 31.1] | 0 | 48.4% [47.1, 49.7] | 0 |

**SDC by bit** (600 faults per bit; full table with intervals in the write-up, section 3):

| Bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F-MNIST SDC % | 16.0 | 21.3 | 22.8 | 24.2 | 26.5 | 30.2 | 32.8 | 32.3 | 41.5 | 51.5 | 58.3 | 69.3 | 90.7 | 69.0 | 90.3 | 77.3 |
| CIFAR-10 SDC % | 24.2 | 29.7 | 36.3 | 37.5 | 41.0 | 46.7 | 53.8 | 63.3 | 72.2 | 79.3 | 88.7 | 91.7 | 96.7 | 93.8 | 78.7 | 95.2 |

Bit 14 is the only bit that gave non-finite outputs: 9.7% of its faults on Fashion-MNIST and 21.3% on CIFAR-10. The rest of bit 14's faults are SDC (0% masked).

**SDC by where the weight sits** (2,400 faults per group: 150 weights × 16 bits):

| Group | Weights in group | Fashion-MNIST SDC | CIFAR-10 SDC | Non-finite (F-MNIST / CIFAR-10) |
|---|---|---|---|---|
| stem + stage 1 | 13,968 / 14,256 | 63.2% | 85.5% | 3 / 16 |
| stage 2 | 50,688 | 59.7% | 75.5% | 1 / 7 |
| stage 3 | 202,752 | 29.2% | 47.1% | 0 / 4 |
| `fc` (final layer) | 640 | 36.5% | 49.1% | 54 / 101 |

**Points to look for in your results** (what surprised us):

1. **Most of our SDC is tiny.** 93% (Fashion-MNIST) and 92% (CIFAR-10) of SDC faults moved accuracy by at most 10 images. On Fashion-MNIST the most common SDC result is accuracy going *up* by one image.
2. **Even mantissa bit 0 causes SDC** (16% and 24%). Week 1's MLP never changed a prediction at bits 0–2. Ours come almost entirely from stem, stage 1 and stage 2 weights. The cause is a few test images whose top two scores are nearly tied: 26 (Fashion-MNIST) and 34 (CIFAR-10) images have a gap below 0.1. Any tiny change can flip one of them, and one flipped image counts as SDC.
3. **The final layer gives most of the non-finite outputs**, while weights near the input give the most SDC. Our guess, which we haven't tested: no batch norm follows `fc`, so a huge weight there goes straight to the output scores. If your VGG has no batch norm, we'd expect more non-finite outputs throughout.

## 2. Exactly what we did, step by step

Everything below is in [`src/resnet20_fault.py`](src/resnet20_fault.py), [`src/bitflip.py`](src/bitflip.py) and [`configs/resnet20.json`](configs/resnet20.json). The decisions and their reasons are in [`plan-resnet20-week2.md`](plan-resnet20-week2.md), section 4 (D1–D9).

### Step 1: Data

- torchvision `FashionMNIST` and `CIFAR10`, the standard splits: 60,000 / 10,000 and 50,000 / 10,000 images.
- No DataLoader. The whole set is loaded as one tensor and pixels are divided by 255.
- **Normalization:** a per-channel mean and standard deviation computed over the whole training set, in float64. Every run records them in `run.json`.
  - Fashion-MNIST: mean 0.2860, std 0.3530.
  - CIFAR-10: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).
- The SHA-256 of the raw dataset files is recorded in `run.json`, and every later run checks it.

### Step 2: Network

- ResNet-20, CIFAR version (He et al., 2016):
  - a 3×3 stem convolution with 16 channels, then batch norm and ReLU;
  - 3 stages of 3 basic blocks with 16, 32 and 64 channels; stages 2 and 3 start with stride 2;
  - each block is conv–BN–ReLU–conv–BN, plus the shortcut, then ReLU;
  - global average pooling, then a linear layer to 10 classes (`fc`).
- Shortcuts are **option A**: identity, or, where the size changes, every second pixel with the new channels padded with zeros. They have no weights.
- Convolutions have no bias. Only `fc` has a bias (10 values).
- He (Kaiming normal) initialization for conv and linear weights. Batch norm and the `fc` bias keep PyTorch's defaults.
- Weights that can be faulted: 268,048 (Fashion-MNIST, 1 input channel) and 268,336 (CIFAR-10, 3 input channels). All parameters: 269,434 and 269,722. The code checks these counts every time it builds the network.

### Step 3: Training (FP32, once per dataset)

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 (applied to all parameters) |
| Learning rate | 0.1, divided by 10 at epochs 80 and 120 |
| Epochs, batch size | 160, 128 (the last batch of each epoch is smaller) |
| Loss | cross-entropy |
| Augmentation | CIFAR-10: random crop after 4-pixel zero padding, and random horizontal flip. Fashion-MNIST: none. |
| Seed | `torch.manual_seed` with a seed derived from master seed 20261104 (see step 6). It drives the initial weights, batch order and augmentation. |
| Hardware | RTX 5090, CUDA, `cudnn.benchmark = True` |
| Time | 6.2 min (Fashion-MNIST), 5.9 min (CIFAR-10) |

GPU training is not bit-repeatable, so each network is trained once and saved. Every later run loads the saved file and checks its SHA-256.

### Step 4: The FP16 reference

1. Convert the whole saved network to FP16 (`model.half()`) and the test images too.
2. Set cuDNN to deterministic (`cudnn.deterministic = True`, `cudnn.benchmark = False`).
3. Evaluate all 10,000 test images **in one batch**.
4. **Check:** FP16 accuracy must be within 0.1 pp (10 images) of FP32. Ours: 0.01 pp and 0.03 pp.
5. **Check:** evaluate the fault-free FP16 network twice. The logits must be byte-identical.
6. The fault-free FP16 predictions are the **reference** that every fault is compared with.
7. **Sanity check:** flip bit 14 of the largest `fc` weight below 1. The outputs must change, and after restoring, the logits must match the reference exactly.

### Step 5: Choosing which weights to fault

- All conv and linear weights are split into **4 groups**: stem + stage 1, stage 2, stage 3, and `fc`. Batch-norm parameters and the `fc` bias are not faulted.
- From each group, draw **150 weights uniformly, without repeats**. Within a group, the tensors are concatenated in parameter order and flat indices are drawn.
- That gives 600 weights × 16 bits = **9,600 faults per dataset**.
- **Why groups instead of Week 1's "150 per weight matrix":** ResNet-20 has 20 weight tensors, and the Fashion-MNIST stem has only 144 weights, so Week 1's rule can't be applied as written. Equal counts per group keep the small early layers from being swamped by stage 3, which holds three-quarters of all weights.
- The chosen weights are written to `chosen_elements.csv`.

### Step 6: Seeds

- Master seed **20261104** (the same as Week 1).
- Each job gets its own stream: `numpy.random.SeedSequence(20261104, spawn_key=(role,))` with roles:

  | Role | Number |
  |---|---|
  | train Fashion-MNIST | 10 |
  | train CIFAR-10 | 11 |
  | one-weight test | 12 |
  | sample stage 1 | 13 |
  | sample stage 2 | 14 |
  | sample stage 3 | 15 |
  | sample `fc` | 16 |

- Sampling uses `numpy.random.default_rng(seed_sequence)`. Torch gets `int(seed_sequence.generate_state(1)[0])`.
- Every role's seed is written to `run.json`.

### Step 7: One fault

For each chosen weight and each bit 0–15:

1. Take a SHA-256 fingerprint of the weight's whole tensor.
2. Flip the bit in place, through an int16 view of the live FP16 tensor (bit 15 = sign, 14–10 = exponent, 9–0 = mantissa).
3. Evaluate all 10,000 test images in one batch, as in step 4.
4. Classify, in this order:
   - **non-finite:** any output score is inf or NaN;
   - **SDC:** all scores finite, and at least one prediction differs from the reference;
   - **masked:** every prediction matches the reference.
5. Restore by writing back the saved original 16 bits (never by flipping again).
6. **Check:** the tensor's fingerprint matches step 1. The run stops if it doesn't.
7. Write one row to `faults.csv`.

### Step 8: Run twice, then compare and summarize

- Run the whole campaign twice from the same saved model. `compare` checks that `chosen_elements.csv` and `faults.csv` are byte-identical. Ours were, on both datasets.
- `summarize` writes `per_bit.csv`, `per_field.csv` and `per_group.csv` with counts, shares and 95% Wilson intervals, plus a per-bit plot.
- Campaign time: 4.2 min (Fashion-MNIST), 5.3 min (CIFAR-10) each.

### Step 9: The one-weight test (a hand check, not part of the counts)

- One weight drawn from all groups, redrawn until |w| < 1, then walked through all 16 bits.
- The new values are checked by hand. For example, Fashion-MNIST `layer3.1.conv1.weight[29070]` = 0.060333251953125; bit 14 makes it 3954.0.

### Commands (from `resnet20/`, on a clean committed tree)

```
.venv\Scripts\python.exe src/resnet20_fault.py train --dataset fashion_mnist --run-name fmnist_train
.venv\Scripts\python.exe src/resnet20_fault.py one-weight --model logs/fmnist_train/model_fp32.pt --run-name fmnist_one_weight
.venv\Scripts\python.exe src/resnet20_fault.py campaign --model logs/fmnist_train/model_fp32.pt --run-name fmnist_campaign_a
.venv\Scripts\python.exe src/resnet20_fault.py campaign --model logs/fmnist_train/model_fp32.pt --run-name fmnist_campaign_b
.venv\Scripts\python.exe src/resnet20_fault.py compare logs/fmnist_campaign_a logs/fmnist_campaign_b
.venv\Scripts\python.exe src/resnet20_fault.py summarize logs/fmnist_campaign_a
```

Then the same with `--dataset cifar10` and `cifar10_` run names. The figures in the write-up come from `.venv\Scripts\python.exe src/make_figures.py`.

**Environment:** Windows 11, RTX 5090, NVIDIA driver 621.01, Python 3.12.10, torch 2.15.0+cu134, torchvision 0.29.2+cu134, numpy 2.5.4, cuDNN 9.26. The exact packages are in [`requirements.txt`](requirements.txt).

## 3. Our output files

The format is the same as Week 1's. If yours match, we can run the same summary and figure code on both.

| File | Columns / contents |
|---|---|
| `faults.csv` | `group, tensor, element_index, bit, original_value, new_value, class, accuracy`: one row per fault; `accuracy` is that fault's test accuracy (4 decimals) |
| `chosen_elements.csv` | `group, tensor, element_index, original_value`, in draw order |
| `per_bit.csv`, `per_field.csv`, `per_group.csv` | `n`, then for each of masked / sdc / non_finite: `count, share, lo, hi` (Wilson 95%) |
| `run.json` | commit, clean-tree flag, versions, GPU, seeds, dataset hashes, normalization, model hash, FP32 and FP16 accuracy, reference logits hash, sanity check, timings |

## 4. What makes the two rows comparable, or not

These are the choices most likely to differ between our groups. Each one changes the numbers.

| Choice | Ours | Why it matters |
|---|---|---|
| Fault model | One bit, one stored FP16 weight, whole network in FP16 | Faulting FP32 weights, or casting to FP16 only at the faulty layer, gives different numbers |
| What can be faulted | Conv and linear weights only; no biases, no batch norm | Biases and BN scales are few but sensitive |
| Sampling | 4 depth groups × 150 weights, uniform within a group | Sampling uniformly over all weights would be dominated by the largest layers (stage 3 here) and lower our SDC share |
| Classification rule | One changed prediction out of 10,000 = SDC; non-finite checked first | With this rule, near-tie test images make even tiny faults count |
| Reference | Fault-free **FP16** predictions, not the true labels | Comparing with labels instead would count differently |
| Test set | All 10,000 images, one batch, cuDNN deterministic | Batch size and convolution algorithm change FP16 rounding |
| Training | He et al. recipe above; normalization from each training set | A less trained or differently scaled network has different near-tie images |
| "Small" vs "large" SDC | Our own split at a 0.1 pp accuracy drop (10 images) | Not part of the doc's definition; we used it to show how much damage SDC does |

## 5. Questions for you

1. Did you use the same fault model and classes (section 4, rows 1, 4 and 5)? If not, what did you change?
2. How did you choose which weights to fault? VGG has more and larger layers. A grouping like ours would be: first conv layers, middle conv layers, last conv layers, classifier, 150 weights each. Tell us what you used and how many weights each group has.
3. Does your VGG use batch normalization? Did you fault biases or batch-norm values?
4. Your training recipe and normalization values, and your FP32 / FP16 accuracy.
5. Did your campaigns give identical results when run twice?
6. Can you share `faults.csv` (or the per-bit, per-field and per-group CSVs) in the format above? With `faults.csv` we can also split your SDC into small and large the same way.
7. Do you record how many predictions changed per fault? We don't. It's an open question for Parnian, because a net accuracy change can hide images that went wrong while others went right.

## 6. Comparison table to fill in

| | ResNet-20, Fashion-MNIST | VGG, Fashion-MNIST | ResNet-20, CIFAR-10 | VGG, CIFAR-10 |
|---|---|---|---|---|
| Faultable weights | 268,048 | | 268,336 | |
| Faults tested | 9,600 | | 9,600 | |
| FP32 / FP16 accuracy | 93.33% / 93.32% | | 91.38% / 91.35% | |
| Masked % | 52.3 | | 34.4 | |
| SDC % | 47.1 | | 64.3 | |
| Non-finite % | 0.6 | | 1.3 | |
| SDC % at bit 0 | 16.0 | | 24.2 | |
| SDC % at bit 14 | 90.3 | | 78.7 | |
| Non-finite % at bit 14 | 9.7 | | 21.3 | |
| Damaging % (drop > 0.1 pp, or non-finite) | 4.1 | | 6.3 | |
| Identical rerun | yes | | yes | |
