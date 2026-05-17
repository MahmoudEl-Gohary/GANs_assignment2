# Conditional Date Generation with Deep Generative Models

## 1. Problem Formulation

**Task.** Given four calendar conditions -- day-of-week, month, leap year, decade -- generate
any valid date (dd-mm-yyyy) satisfying all conditions simultaneously. This is a conditional
generation problem with multiple correct outputs per input.

**Input representation.** Each condition is encoded as a categorical index:
- Day-of-week: 7 classes (MON=0, ..., SUN=6)
- Month: 12 classes (JAN=0, ..., DEC=11)
- Leap year: 2 classes (False=0, True=1)
- Decade: 41 classes ([180]=0, ..., [220]=40)

These four indices form the input tensor of shape (4,), embedded via separate `nn.Embedding` layers.

**Output tokenization.** The target date is tokenized character-by-character.
Two key design decisions:

1. **Reordering**: Dates are converted from `dd-mm-yyyy` to `yyyy-mm-dd` so the model
   predicts the most constrained digits first (year digits are tightly bound by the decade
   condition; day digits require complex calendar reasoning).
2. **Zero-padding**: Day and month are padded to 2 digits (e.g., `1` -> `01`), producing a
   **fixed-length** 12-token sequence: `<SOS> y y y y - m m - d d <EOS>`. This eliminates
   variable-length ambiguity. The vocabulary is 14 tokens: `{<PAD>, <SOS>, <EOS>, -, 0-9}`.

**Loss functions.**
- LSTM, Transformer: Cross-entropy loss with `ignore_index=<PAD>` (teacher-forced training).
- GAN: Binary cross-entropy (Discriminator classifies real vs. fake sequences).
- CVAE: Reconstruction cross-entropy + KL divergence (weighted at 0.01 to prioritize reconstruction).

---

## 2. Architectures

### 2.1 SimpleLSTMGenerator

A 2-layer autoregressive LSTM decoder. Condition embeddings are:
1. Projected via linear layers to initialize both h0 and c0 (per layer).
2. **Concatenated to the character embedding at every decoder step**, ensuring persistent
   access to all conditions (especially day-of-week) throughout generation.

LSTM input dimension = `emb_dim (64) + cond_dim (256)` = 320 per step.

### 2.2 SmallTransformerGenerator

An encoder-decoder Transformer (3 layers, 4 heads, d_model=128). The 4 condition
embeddings form a length-4 encoder source sequence for cross-attention. Additionally,
a projected condition bias vector is added to every decoder position embedding, providing
a dual injection pathway (cross-attention + additive bias).

### 2.3 Conditional GAN

A standard conditional GAN with Gumbel-Softmax for discrete text generation. The Generator
takes noise + condition embeddings through a 3-layer MLP with LayerNorm, outputting soft
one-hot vectors. The Discriminator receives flattened one-hot sequences + condition embeddings.
Label smoothing (0.9/0.1) is applied to stabilize training.

### 2.4 Conditional VAE

An MLP-based conditional VAE. The encoder maps (target sequence + conditions) to a 64-dim
latent space via reparameterization. The decoder reconstructs the sequence from a latent
sample + conditions. At inference, z is sampled from N(0, I).

---

## 3. Evaluation Metric

Standard accuracy is inappropriate because multiple outputs are correct per input.
Instead, I use **calendar validation accuracy**: a generated date passes only if it
simultaneously satisfies all four conditions (valid calendar date + correct weekday +
correct month + correct leap year status + correct decade). This is implemented in
`model/utils.py:validate_date()` using Python's `datetime` module.

Additionally, I track **per-condition accuracy** to diagnose which conditions the models
struggle with. This breakdown proved critical for understanding model behavior (Section 5).

---

## 4. Results

### 4.1 Per-Condition Accuracy Breakdown (10,000 samples)

| Model              | All OK | Valid Date | Day   | Month  | Leap   | Decade  |
|--------------------|--------|-----------|-------|--------|--------|---------|
| Simple LSTM        | 15.00% | 100.00%   | 15.00%| 100.00%| 100.00%| 100.00% |
| Small Transformer  | 13.85% | 100.00%   | 13.85%| 100.00%| 100.00%| 100.00% |
| Conditional GAN    | 13.39% | 99.27%    | 13.72%| 98.67% | 98.09% | 98.24%  |
| Conditional VAE    | 14.05% | 99.20%    | 14.12%| 99.19% | 98.78% | 99.07%  |

### 4.2 Sample Predictions

**LSTM (best model):**

| Conditions | Predicted | Target | Status |
|---|---|---|---|
| [SUN] [JUN] [False] [188] | 15-6-1882 | 10-6-1883 | FAIL (wrong weekday) |
| [FRI] [MAY] [False] [214] | 15-5-2142 | 11-5-2142 | FAIL (wrong weekday) |
| [WED] [FEB] [False] [193] | 15-2-1933 | 15-2-1939 | OK |

**Conditional VAE (most diverse outputs):**

| Conditions | Predicted | Target | Status |
|---|---|---|---|
| [THU] [SEP] [False] [201] | 21-9-2017 | 27-9-2018 | OK |
| [SAT] [SEP] [False] [181] | 7-9-1811 | 2-9-1815 | OK |
| [TUE] [SEP] [False] [199] | 31-9-1998 | 8-9-1998 | FAIL (invalid: Sep has 30 days) |

---

## 5. Analysis and Reflection

### 5.1 Key Finding: Day-of-Week is the Sole Bottleneck

The per-condition breakdown reveals a striking pattern: **all four models achieve near-perfect
accuracy on month, leap year, and decade conditions**, but hover around **~14% on day-of-week**
-- which is exactly 1/7 (random chance).

This is because:
- **Month, decade, leap year** are *directly readable* from the date digits. Given `yyyy-mm-dd`,
  the decade is simply the first 3 digits of the year, the month is digits 5-6, and leap year
  is a simple function of the year.
- **Day-of-week** requires computing the equivalent of **Zeller's congruence** -- a complex
  modular arithmetic function involving the day, month, century, and year-within-century.
  This is fundamentally difficult for neural networks operating at the character level.

### 5.2 Mode Collapse on Day-of-Month

The LSTM always generates day=15, and the Transformer always generates day=23. Both models
have found a "safe" strategy: pick a fixed day-of-month that (a) is valid for all months
(both <= 28), and (b) produces a correct weekday ~14% of the time by chance. The models
are not truly generating -- they are memorizing a single safe day value.

The GAN and VAE produce more diverse day values (no mode collapse), but their day-of-week
accuracy is statistically identical (~14%), confirming that diversity does not translate to
correctness for this condition.

### 5.3 What Worked

- **Token reordering** (yyyy-mm-dd): Ensured deterministic conditions (decade) are predicted
  first, yielding 100% decade accuracy.
- **Zero-padding**: Fixed-length sequences eliminated padding-induced positional errors.
  Prior to this fix, GAN and VAE produced structurally invalid outputs (e.g., `513-2-2191`).
- **Condition injection at every step** (LSTM): Prevented the model from "forgetting"
  conditions during autoregressive generation, though it did not solve the weekday problem.

---

## 6. Best Practices Applied

| Practice | Implementation |
|----------|----------------|
| Manual seed | `torch.manual_seed(42)` in all trainers |
| Train/test split | 90/10 split with `Generator().manual_seed(SEED)` |
| Data shuffling | `shuffle=True` in training DataLoaders |
| LR scheduler | `ReduceLROnPlateau` monitors validation loss |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |
| Best-model checkpoint | Saves weights only when validation accuracy improves |
| Type hinting | All function signatures are type-annotated |
| Modular code structure | Separate files: `dataset.py`, `models.py`, `train.py`, `evaluate.py`, `predict.py`, `utils.py` |
| Reproducibility | Fixed seeds, deterministic splits, documented hyperparameters |
