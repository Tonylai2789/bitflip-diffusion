# ResNet-20 bit-flip study: fault-model paragraph and short answers (DRAFT)

> **DRAFT for Tony to edit. Not for hand-in as is.** Claude wrote this on Nov 21, 2026. The doc asks for the paragraph and the short answers "in your own words" (plan, section 0, rule 6), so please rewrite them in your own words. Every number was checked against the official logs from commit `27ee894` (`logs/fmnist_*`, `logs/cifar10_*`) and against [`results-2026-11-20.md`](results-2026-11-20.md). No `dev_*` logs were used. Nothing here is committed.

**Which questions these answer.** The Google Doc's Week2 tab asks no new questions. It only gives the 2 × 2 grid, the VGG description and the reproducibility rules. So this draft answers the four Week 1 short-answer questions (Proposal tab, "What to hand in by the end of the week") for ResNet-20 on both datasets. The paragraph follows step 7's checklist: what was flipped, where, which bits, how many, how they were chosen, how randomness was controlled, how values were restored, and how faults were classified.

**How to read the answers.** Plain sentences are results from the logs. Sentences marked **Not tested:** are possible explanations that we did not check. Each answer names the section, figure or table of `results-2026-11-20.md` it comes from. A few numbers are not in the results file; for those, the answer names the log file they were counted from.

## 1. Fault model

We studied single-bit faults in the stored weights of ResNet-20 (the CIFAR version from He et al., 2016), trained once on Fashion-MNIST and once on CIFAR-10. The network has a 3 × 3 convolution with 16 filters (the "stem"), then 3 stages of 3 residual blocks with 16, 32 and 64 filters, batch normalization after every convolution, shortcuts that pad with zeros and have no weights, average pooling, and a final linear layer from 64 to 10 (`fc`). That makes 20 weight tensors (19 convolutions and `fc.weight`): 268,048 weights for Fashion-MNIST (1 input channel) and 268,336 for CIFAR-10 (3 channels). Each network was trained in FP32 on an RTX 5090 GPU, with SGD (momentum 0.9, weight decay 1e-4), learning rate 0.1 divided by 10 at epochs 80 and 120, 160 epochs and batch size 128. CIFAR-10 images got a random crop (4-pixel padding) and a random horizontal flip; Fashion-MNIST images were not augmented. Inputs were scaled with the per-channel mean and standard deviation of each training set (Fashion-MNIST: 0.2860 and 0.3530; CIFAR-10: (0.4914, 0.4822, 0.4465) and (0.2470, 0.2435, 0.2616)). GPU training is not bit-for-bit repeatable, so we trained once per dataset, saved the weights with their SHA-256, and loaded that file for every test. For the faults, the whole network was converted to FP16 and run on the same GPU with cuDNN set to deterministic. FP16 accuracy was within 0.1 percentage points of FP32 (Fashion-MNIST 93.32% vs 93.33%, CIFAR-10 91.35% vs 91.38%). Faults went only into the convolution and linear weights. Biases and all batch-norm values were not faulted. We split the weights into 4 groups: stem + stage 1, stage 2, stage 3, and `fc.weight`. From each group we picked 150 weights uniformly at random from all the weights in that group, with no repeats, 600 in total. For each weight we flipped each of its 16 bits (0–15: bit 15 is the sign, bits 14–10 the exponent, bits 9–0 the mantissa) one at a time, giving 600 × 16 = 9,600 faults per dataset. Each fault was a single bit flipped directly in the weight's FP16 bits in the network's memory, with only one fault present at a time. It stayed in place while the network classified all 10,000 test images of that dataset. We then classified the fault against the fault-free FP16 network's predictions: non-finite if any output score was infinity or NaN; otherwise silent data corruption (SDC) if at least one of the 10,000 predictions changed; otherwise masked. To restore, we wrote the weight's saved original bits back. A SHA-256 fingerprint of the whole weight tensor, taken before the flip and after the restore, had to match, or the run would stop. All randomness came from one master seed, 20261104, which numpy's `SeedSequence` split into separate streams by role number: 10 for training on Fashion-MNIST and 11 for training on CIFAR-10 (starting weights, batch order and augmentation), 12 for picking the single weight walked through all 16 bits (drawn from all weights, redrawn until its size was below 1; it was `layer3.1.conv1.weight` index 29070 for Fashion-MNIST and index 28942 for CIFAR-10), and 13–16 for sampling the four groups. Evaluation used no randomness and always used the same 10,000 test images. Two complete runs with the same seed produced byte-identical result files on both datasets.

Sources: results file, [Setup](results-2026-11-20.md#setup), [section 1](results-2026-11-20.md#1-the-trained-networks), [section 2](results-2026-11-20.md#2-identical-reruns) and [section 7](results-2026-11-20.md#7-one-weight-through-all-16-bits); `configs/resnet20.json`; every `run.json` (commit `27ee894`, clean tree).

## 2. Short answers

### 2.1 Which bit positions cause silent data corruption and which cause non-finite outputs, with your explanation from the way FP16 stores numbers

FP16 stores a number in 16 bits as

$$\text{value} = (-1)^{s} \cdot 2^{E-15} \cdot \left(1 + \frac{M}{1024}\right)$$

where $s$ is the sign bit (bit 15), $E$ is the 5-bit exponent (bits 14–10) and $M$ is the 10-bit mantissa (bits 9–0). Flipping the sign makes the weight negative (or positive). Flipping an exponent bit multiplies or divides the weight by 65,536 (bit 14), 256 (bit 13), 16 (bit 12), 4 (bit 11) or 2 (bit 10). It multiplies if that bit was 0 and divides if it was 1. Flipping a mantissa bit changes the weight by a fraction of itself, which doubles with each bit: about 0.05–0.1% for bit 0, up to 25–50% for bit 9.

All 600 weights on each dataset were smaller than 2 (largest 1.44 on Fashion-MNIST and 1.60 on CIFAR-10). So bit 14 was 0 in every weight, and flipping it always multiplied by 65,536.

**Every bit position caused SDC on both datasets.** Only bit 14 caused non-finite outputs.

| Bit | Field | What the flip did (count of weights: F-MNIST / CIFAR-10) | F-MNIST SDC | CIFAR-10 SDC | F-MNIST non-finite | CIFAR-10 non-finite |
|---|---|---|---|---|---|---|
| 0 | mantissa | changed it by 0.05–0.1% | 16.0% | 24.2% | 0 | 0 |
| 1 | mantissa | 0.1–0.2% | 21.3% | 29.7% | 0 | 0 |
| 2 | mantissa | 0.2–0.4% | 22.8% | 36.3% | 0 | 0 |
| 3 | mantissa | 0.4–0.8% | 24.2% | 37.5% | 0 | 0 |
| 4 | mantissa | 0.8–1.6% | 26.5% | 41.0% | 0 | 0 |
| 5 | mantissa | 1.6–3.1% | 30.2% | 46.7% | 0 | 0 |
| 6 | mantissa | 3.1–6.3% | 32.8% | 53.8% | 0 | 0 |
| 7 | mantissa | 6.3–12.5% | 32.3% | 63.3% | 0 | 0 |
| 8 | mantissa | 12.5–25% | 41.5% | 72.2% | 0 | 0 |
| 9 | mantissa | 25–50% | 51.5% | 79.3% | 0 | 0 |
| 10 | exponent | ×2 (290 / 305) or ÷2 (310 / 295) | 58.3% | 88.7% | 0 | 0 |
| 11 | exponent | ×4 (274 / 299) or ÷4 (326 / 301) | 69.3% | 91.7% | 0 | 0 |
| 12 | exponent | ×16 (388 / 320) or ÷16 (212 / 280) | 90.7% | 96.7% | 0 | 0 |
| 13 | exponent | ×256 (39 / 30) or ÷256 (561 / 570) | 69.0% | 93.8% | 0 | 0 |
| 14 | exponent | ×65,536 (all 600) | 90.3% | 78.7% | 9.7% | 21.3% |
| 15 | sign | became its negative | 77.3% | 95.2% | 0 | 0 |

Each bit has 600 faults (150 per group). The SDC and non-finite shares, with their 95% intervals, are in the per-bit table in [section 3](results-2026-11-20.md#3-what-each-bit-did) (Figure 2) and in Figure 3 ([section 4](results-2026-11-20.md#4-sdc-by-bit-compared-with-week-1)). The ×/÷ counts were counted from the original values in `logs/*_campaign_a/chosen_elements.csv`. The mantissa ranges hold for normal FP16 numbers, which is all but 1 (Fashion-MNIST) and 2 (CIFAR-10) of the 600 weights.

**The mantissa (bits 0–9).** SDC grows with the bit, from 16% (Fashion-MNIST) and 24% (CIFAR-10) at bit 0 to 52% and 79% at bit 9, because each bit makes twice the change of the one below it. But this SDC is small. Not one mantissa fault dropped accuracy by more than 0.1 points (10 images) ([section 3](results-2026-11-20.md#3-what-each-bit-did), Figure 2).

**The exponent (bits 10–13).** These bits multiply or divide the weight instead of nudging it, so they cause more SDC: 58–91% on Fashion-MNIST and 89–97% on CIFAR-10. Making a weight bigger caused SDC more often than making it smaller (counted from `logs/*_campaign_a/faults.csv`):

| Bit | F-MNIST SDC when multiplied | F-MNIST SDC when divided | CIFAR-10 SDC when multiplied | CIFAR-10 SDC when divided |
|---|---|---|---|---|
| 10 | 188 of 290 | 162 of 310 | 277 of 305 | 255 of 295 |
| 11 | 216 of 274 | 200 of 326 | 282 of 299 | 268 of 301 |
| 12 | 379 of 388 | 165 of 212 | 316 of 320 | 264 of 280 |
| 13 | 36 of 39 | 378 of 561 | 30 of 30 | 533 of 570 |

That explains the peak at bit 12 and the dip at bit 13. Bit 12 multiplies most weights by 16, and those faults were SDC 98% (Fashion-MNIST) and 99% (CIFAR-10) of the time. Bit 13 divides most weights by 256, which nearly erases the weight instead of enlarging it. Week 1's network had the same peak and dip ([section 4](results-2026-11-20.md#4-sdc-by-bit-compared-with-week-1), Figure 3).

**Bit 14** is different from every other bit. It never masked a fault. Its SDC is often large: the median accuracy drop was 6.44 points on Fashion-MNIST and 1.00 on CIFAR-10, and the worst left accuracy at 4.6% and 8.1%. Bit 14 caused 93% (Fashion-MNIST) and 79% (CIFAR-10) of all damaging faults, meaning an accuracy drop of more than 0.1 points or a non-finite output ([section 3](results-2026-11-20.md#3-what-each-bit-did)).

**The sign bit (15)** caused SDC in 77% and 95% of faults, almost all of it small ([section 3](results-2026-11-20.md#3-what-each-bit-did)). Only 10 of 464 and 31 of 571 sign-bit SDCs dropped accuracy by more than 0.1 points (counted from `faults.csv`).

**Non-finite outputs came only from bit 14:** 58 of 600 faults (9.7%) on Fashion-MNIST and 128 of 600 (21.3%) on CIFAR-10 ([section 3](results-2026-11-20.md#3-what-each-bit-did)). Two things happened (counted from `faults.csv`):

- **The weight itself stopped being a number.** 4 weights on Fashion-MNIST and 5 on CIFAR-10 were 1 or larger, all in `fc`. Their exponent was already 15, so flipping bit 14 set it to 31, which FP16 uses for infinity and NaN. Each of these weights became NaN, and each fault was non-finite.
- **The weight stayed finite, but the output didn't.** In the other 54 (Fashion-MNIST) and 123 (CIFAR-10) non-finite faults, the flipped weight was a large but finite number: the largest was 63,936, under FP16's limit of 65,504. So the infinity or NaN appeared later, while the network computed with that weight. The size of the flipped weight alone doesn't decide this. In `fc` on Fashion-MNIST, non-finite faults had flipped weights from 17,104 up, while faults that stayed finite had flipped weights up to 28,160.

Most non-finite faults were in `fc`: 54 of 58 on Fashion-MNIST and 101 of 128 on CIFAR-10 ([section 5](results-2026-11-20.md#5-where-in-the-network-the-weight-sits), Figure 4). The `fc` weights are also larger than the convolution weights: their median size was 0.23 (Fashion-MNIST) and 0.35 (CIFAR-10), against 0.03–0.10 in the three convolution groups (from `chosen_elements.csv`). So multiplying them by 65,536 gives bigger numbers.

**Not tested:** that the overflow happens when the huge weight is multiplied by its input and the result passes 65,504, as Week 1 showed for its network. We didn't log the values inside the network, so we can't check it here. **Not tested:** that `fc` gives the most non-finite outputs because no batch-norm layer follows it, so the huge value goes straight into a class score.

### 2.2 Which bit positions never changed any prediction and why

**None, on either dataset.** Every bit position changed at least one prediction in some faults. The closest is bit 0, the lowest mantissa bit: 84.0% of its faults were masked on Fashion-MNIST and 75.8% on CIFAR-10, so 16.0% [13.3, 19.1] and 24.2% [20.9, 27.7] were SDC ([section 3](results-2026-11-20.md#3-what-each-bit-did), per-bit table). This is the biggest difference from Week 1, where bits 0–2 never changed a prediction ([section 4](results-2026-11-20.md#4-sdc-by-bit-compared-with-week-1), Figure 3 and table).

What we found about these low-bit SDCs:

- **They are tiny.** For bits 0–2, every SDC fault changed accuracy by at most 2 images, net. On Fashion-MNIST, all 361 of them raised accuracy by exactly one image (counted from `faults.csv`). The logs record accuracy only, not how many predictions changed, so this is the net change (open question 5 in the plan).
- **They come from the layers near the input.** Of the bits 0–2 SDCs, 200 + 156 of 361 on Fashion-MNIST and 299 + 182 of 541 on CIFAR-10 were in stem + stage 1 and stage 2. Stage 3 and `fc` gave almost none ([section 6](results-2026-11-20.md#6-why-even-the-lowest-bit-causes-sdc), table "for bits 0–2").
- **Both networks have borderline test images.** These are images whose top two output scores are almost equal: 26 (Fashion-MNIST) and 34 (CIFAR-10) have a gap below 0.1, and one CIFAR-10 image is an exact tie. One flipped image is enough to make a fault SDC ([section 6](results-2026-11-20.md#6-why-even-the-lowest-bit-causes-sdc), Figure 5).
- **Borderline images alone don't explain the difference from Week 1.** Week 1's network had a similar median gap (12.3) and even more images below 0.01 (6, against 1 and 3 here) ([section 6](results-2026-11-20.md#6-why-even-the-lowest-bit-causes-sdc)).

**Not tested:** why a tiny change to an early weight reaches the borderline images. One possible reason: a convolution weight in the stem or stage 1 is applied at every one of the 784 (Fashion-MNIST) or 1,024 (CIFAR-10) pixel positions of every image, so even a 0.05% change nudges every output score a little. An `fc` weight, by contrast, feeds only one class score. **Not tested:** whether the bit-0 SDCs on Fashion-MNIST all flip the same image.

### 2.3 How the result would change if the network were stored in 32-bit floating point

We did not run an FP32 campaign, so this answer is reasoning only. (Week 1 had a quick FP32 check; Week 2 does not.)

FP32 has 1 sign bit (31), 8 exponent bits (30–23) and 23 mantissa bits (22–0). The campaign would have 600 × 32 = 19,200 faults per dataset. For weights that are normal FP16 numbers (all but 1 / 2 of the 600), FP32 bits line up with FP16 bits like this:

| FP32 bits | Same as | What the flip does to our weights |
|---|---|---|
| 31 (sign) | FP16 bit 15 | Same: the weight changes sign |
| 30 | FP16 bit 14 | ×2¹²⁸ for all 600 weights, because all are below 2 |
| 27–29 | nothing in FP16 | ÷2¹⁶, ÷2³² or ÷2⁶⁴: the weight is basically erased. This holds for every weight, except that bit 27 multiplies by 2¹⁶ for the one CIFAR-10 weight below 2⁻¹⁵ |
| 23–26 | FP16 bits 10–13 | Same ×/÷ by 2, 4, 16, 256, in the same direction for the same weights (apart from rare rounding cases) |
| 13–22 | FP16 bits 0–9 | Same fraction of the weight |
| 0–12 | nothing in FP16 | Smaller changes than FP16 bit 0: from about 0.000006% (bit 0) to 0.05% (bit 12) |

What we expect (**none of this is measured**):

- **Bits 23–26 and 13–22 and the sign** should behave much like their FP16 partners in the per-bit table ([section 3](results-2026-11-20.md#3-what-each-bit-did)). They won't match exactly, because the reference network and all the arithmetic would be FP32 too.
- **Bit 30 should behave like FP16 bit 14.** In both formats the top exponent bit multiplies a small weight by about the format's largest number (65,536 against a limit of 65,504; 2¹²⁸ against a limit of about 3.4 × 10³⁸). So in both, the same weights (1 or larger) become infinity or NaN, and a smaller weight overflows at about the same input size. Week 1's quick FP32 check found exactly this for its network: bit 30 gave the same 181 SDC and 99 non-finite faults as FP16 bit 14.
- **Bits 27–29** are new. They nearly erase the weight, like FP16 bit 13 does when it divides by 256. Those bit-13 faults were SDC 67% (Fashion-MNIST, 378 of 561) and 94% (CIFAR-10, 533 of 570) of the time (from `faults.csv`). So we expect bits 27–29 to give a lot of SDC too.
- **Bits 0–12** add 13 low bits with very small effects. That would pull the overall SDC share down. But unlike Week 1, we can't say these bits would all be masked. Even FP16 bit 0 gave 16% and 24% SDC, and SDC falls only slowly from bit 2 to bit 0 (22.8% → 16.0% on Fashion-MNIST, 36.3% → 24.2% on CIFAR-10). We'd need a run to know where it reaches zero.
- **Non-finite outputs** should still come only from the top exponent bit (bit 30), so their share of all faults would roughly halve, because there would be twice as many faults.

### 2.4 Which parts of your fault model you expect to need changing for a diffusion model, and why

Week 1's answer still holds. The fault itself and the way we inject and restore it can stay the same. Much of the setup must change because the diffusion model is bigger, is stored in FP32, and makes images instead of labels. Week 2 adds these lessons:

| Part | What changes | Why (from Week 2) |
|---|---|---|
| Sampling groups | Sample by UNet region, report each region separately, and decide whether overall totals weigh regions equally or by size | SDC ranged from 29% to 86% between our 4 groups. Our totals give each group equal weight, but stage 3 holds 76% of the weights and only 25% of the faults ([section 5](results-2026-11-20.md#5-where-in-the-network-the-weight-sits), [section 8](results-2026-11-20.md#8-limits)) |
| Normalization layers | Decide whether to fault the normalization parameters. We left batch norm out (decision D3); the DDPM UNet uses group normalization, with a learned scale and shift | It's a choice we had to make here too, and it changes what counts as "a weight" |
| SDC definition | Measure how big the change is, not only whether anything changed | 92–93% of our SDC moved accuracy by 0.1 points or less, and even bit 0 counted as SDC because of borderline images ([section 3](results-2026-11-20.md#3-what-each-bit-did), [section 6](results-2026-11-20.md#6-why-even-the-lowest-bit-causes-sdc)). A generated image has thousands of pixel values, so almost any fault would probably change some of them (expected, not tested) |
| Which bits, if time is short | Keep bit 14 and the sign bit; a few mantissa bits can show the small-SDC trend | Bit 14 caused 93% / 79% of the damaging faults; no mantissa fault dropped accuracy by more than 0.1 points ([section 3](results-2026-11-20.md#3-what-each-bit-did)) |
| Number format | Decide FP32 or FP16, and check FP16 by image quality | Converting to FP16 cost at most 0.03 points here ([section 1](results-2026-11-20.md#1-the-trained-networks)), but a generator has no accuracy to check |
| Repeatable runs | Deterministic GPU settings, and fault-free images from the same seeded noise as the reference | Byte-identical reruns here needed cuDNN set to deterministic (decision D8; [section 2](results-2026-11-20.md#2-identical-reruns)) |
| Timing | Persistent (all denoising steps) vs only some steps | The faulty weight is used again in every denoising step (Week 1's point; research question 2) |

## 3. Notes for Tony (not part of the answers)

- **Please rewrite in your own words.** The doc asks for that, and the plan says you write these texts.
- **Questions used:** the four Week 1 short-answer questions, because the Week2 tab has none. If Parnian expects Week 2-specific questions (for example, Fashion-MNIST vs CIFAR-10, or the VGG comparison in decision D9), they aren't covered here. The VGG comparison is left out on purpose: their results don't exist yet.
- **Numbers that are new here, not in the results file:** the ×/÷ split for bits 10–13, the NaN weights from bit 14, the flipped-weight ranges, the net change of bits 0–2 SDCs, the sign-bit damaging counts and the `fc` weight sizes. All were counted from `logs/{fmnist,cifar10}_campaign_a/faults.csv` and `chosen_elements.csv`. They agree with the results file wherever the two overlap.
- **A small imprecision in the results file:** "How to read this report" says bit 14 "always multiplied the weight by 65,536". For the 4 / 5 `fc` weights of size 1 or more, the result can't be stored, and the weight became NaN. The numbers in the results file are unaffected. I did not edit it.
- **The stem got few draws.** Group 1 draws uniformly from all stem + stage 1 weights, so the stem (144 or 432 weights) got 1 of 150 draws on Fashion-MNIST and 3 of 150 on CIFAR-10 (`chosen_elements.csv`). So "stem + stage 1" results are really about stage 1.
- **Answer 2.3 has no data behind it.** If you want data like Week 1's, a one-off FP32 campaign of the same 600 weights would settle it.
