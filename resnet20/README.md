# ResNet-20 bit-flip study (Week 2, Nov 10 – Nov 16, 2026)

A short guide for people. The full step-by-step plan for AI agents, with every decision and a work log, is [`plan-resnet20-week2.md`](plan-resnet20-week2.md).

**Status (Nov 18):** the official runs are done for both datasets, from committed code, and every check passed. Test accuracy in FP16: Fashion-MNIST 93.32%, CIFAR-10 91.35%. Each 9,600-fault test gave identical results both times. Out of 9,600 faults, answers changed silently (SDC) in 4,525 on Fashion-MNIST and 6,172 on CIFAR-10. Details are in the plan's work log.

## What we're doing

Week 1 flipped single bits in the weights of a small digit-recognition network (MNIST). This week repeats that on a deeper network, **ResNet-20**, trained on two image datasets:

- **Fashion-MNIST:** 28 × 28 grayscale pictures of clothing, 10 classes.
- **CIFAR-10:** 32 × 32 color photos (planes, cars, birds, …), 10 classes.

Another group does the same with a small VGG-style network, so together we fill a 2 × 2 grid (two networks × two datasets).

ResNet-20 matters because it is built like the diffusion models this project is really about: convolution layers, "skip connections" that carry a signal around each block, and batch normalization. We want to see whether those parts soak up a bit flip or spread it.

## The test, in short

Same rules as Week 1:

1. Train the network once and save it. Store its weights as 16-bit floating point numbers (FP16).
2. Pick weights at random. For each one, flip each of its 16 bits, one at a time, and check the network's answers on all 10,000 test images.
3. Sort each test into **non-finite** (infinity or NaN: visibly broken), **silent data corruption** (normal-looking outputs, at least one answer changed) or **masked** (no answer changed).
4. Put the bit back and check the weight is exactly as before.

## Choices we made (settled Nov 17)

- Our group runs ResNet-20 on both datasets.
- Weights are drawn from four groups (first layer + stage 1, stage 2, stage 3, final layer), 150 per group: 600 weights × 16 bits = 9,600 tests per dataset. **Change from Week 1, and why:** Week 1's rule was 150 per weight matrix, but ResNet-20 has 20 weight tensors and the Fashion-MNIST first layer has only 144 weights, so the rule can't be applied as written. Grouping by stage keeps equal counts per group, so the small early layers aren't hidden by the large late ones.
- Batch-normalization numbers are left out, like biases in Week 1. **Why:** Week 1 faults weights only; the learned batch-norm values are 1,376 numbers, about 0.5% of the 269,722 parameters (counted by hand; to be checked against the model code).
- Master seed 20261104 (the same as Week 1), with separate random streams for training and for choosing which weights to fault.
- The standard ResNet-20 training recipe: 160 epochs, as in the original paper.
- Image normalization values computed from each training set and written down with every run.
- FP16 accuracy must be within 0.1 points of FP32. Every fault campaign runs twice and must give identical results.

## Setup (Windows RTX 5090 PC)

This folder has its own virtual environment. From `resnet20/`:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu134
.venv\Scripts\python.exe -m pytest tests/ -v -rs
```

Datasets download into `data/` (git-ignored) through torchvision.
