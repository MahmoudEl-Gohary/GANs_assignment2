# Dates Generator -- Conditional Date Generation with Deep Generative Models

## Overview

This project implements **four neural network architectures** for conditional date generation.
Given a set of conditions (day-of-week, month, leap year, decade), each model generates a
valid calendar date satisfying those conditions.

## Models

| # | Model | Type | Source |
|---|-------|------|--------|
| 1 | **SimpleLSTMGenerator** | Autoregressive LSTM | Course (RNN-based) |
| 2 | **SmallTransformerGenerator** | Encoder-Decoder Transformer | Course (Attention-based) |
| 3 | **Conditional GAN** | Generator + Discriminator | Required (GAN) |
| 4 | **Conditional VAE** | Variational Autoencoder | Extra (VAE) |

## Repository Structure

```
dates_generator/
├── data/
│   ├── data.txt                 # Full dataset (146k samples)
│   ├── example_input.txt        # Example input (conditions only)
│   └── example_output.txt       # Generated output
├── model/
│   ├── __init__.py
│   ├── dataset.py               # Tokenizer and Dataset classes
│   ├── models.py                # All four model architectures
│   ├── train.py                 # Training scripts with CLI
│   ├── evaluate.py              # Evaluation with per-condition breakdown
│   ├── predict.py               # Inference CLI (assignment interface)
│   ├── utils.py                 # Shared validation utilities
│   └── weights/                 # Trained model weights (.pth)
├── environment.yml              # Conda environment specification
├── pyproject.toml               # Project configuration (uv)
└── README.md
```

## Setup

```bash
# Option 1: Using uv (recommended)
uv sync

# Option 2: Using conda
conda env create -f environment.yml
conda activate dates-generator
```

## Usage

### Training

```bash
# Train a specific model
python -m model.train --model lstm --epochs 30
python -m model.train --model transformer --epochs 30
python -m model.train --model gan --epochs 30
python -m model.train --model cvae --epochs 30

# Train all models
python -m model.train --model all --epochs 30
```

### Inference (Assignment Interface)

```bash
# Default model (lstm)
python predict.py -i data/example_input.txt -o data/example_output.txt

# Specify a model
python predict.py -i data/example_input.txt -o data/example_output.txt --model transformer
```

### Evaluation

```bash
python -m model.evaluate
```

## Design Decisions

- **Token ordering**: Dates are internally reordered from `dd-mm-yyyy` to `yyyy-mm-dd`
  so the model predicts the most constrained digits (year/decade) first.
- **Zero-padded encoding**: Day and month are zero-padded to 2 digits, yielding a
  fixed 12-token sequence (`<SOS>yyyy-mm-dd<EOS>`) that eliminates positional ambiguity.
- **Condition injection**: The LSTM concatenates condition embeddings to the character
  embedding at every decoder step. The Transformer adds a projected condition bias to
  every decoder position in addition to cross-attention.
- **Evaluation metric**: Calendar validation accuracy -- a generated date must satisfy
  all four conditions (day-of-week, month, leap year, decade) to count as correct.
- **Reproducibility**: `torch.manual_seed(42)` is set in every training function.
