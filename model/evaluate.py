"""Evaluate all trained models and report per-condition accuracy breakdown.

Usage:
    python -m model.evaluate
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from model.dataset import DatesTokenizer, DatesDataset
from model.models import (
    SimpleLSTMGenerator,
    SmallTransformerGenerator,
    Generator,
    ConditionalVAE,
)
from model.utils import validate_date, check_individual_conditions

WEIGHTS_DIR = Path("model/weights")
DATA_PATH = Path("data/data.txt")
EVAL_SAMPLES = 10_000
SEED = 42


def evaluate_models() -> None:
    """Load all models, evaluate on a fixed subset, and print results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}\n")

    tokenizer = DatesTokenizer()
    dataset = DatesDataset(DATA_PATH, tokenizer)
    vocab_size = tokenizer.vocab_size

    # Fixed random subset for consistent evaluation
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:EVAL_SAMPLES].tolist()
    eval_dataset = Subset(dataset, indices)
    dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    # Model registry
    models: dict[str, dict] = {
        "Simple LSTM": {
            "model": SimpleLSTMGenerator(vocab_size=vocab_size).to(device),
            "weights": WEIGHTS_DIR / "lstm_weights.pth",
            "type": "indices",
        },
        "Small Transformer": {
            "model": SmallTransformerGenerator(vocab_size=vocab_size).to(device),
            "weights": WEIGHTS_DIR / "transformer_weights.pth",
            "type": "indices",
        },
        "Conditional GAN": {
            "model": Generator(vocab_size=vocab_size).to(device),
            "weights": WEIGHTS_DIR / "gan_generator_weights.pth",
            "type": "logits",
        },
        "Conditional VAE": {
            "model": ConditionalVAE(vocab_size=vocab_size).to(device),
            "weights": WEIGHTS_DIR / "cvae_weights.pth",
            "type": "logits",
        },
    }

    # Header
    print(f"{'Model':<20} | {'All OK %':>8} | {'Valid %':>8} | "
          f"{'Day %':>7} | {'Month %':>8} | {'Leap %':>7} | {'Decade %':>9} | {'Samples':>8}")
    print("-" * 105)

    for name, config in models.items():
        model = config["model"]
        weights_path: Path = config["weights"]

        if not weights_path.exists():
            print(f"{name:<20} | {'N/A':>8} | Weights not found at {weights_path}")
            continue

        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True), strict=False
        )
        model.eval()

        # Counters
        total = 0
        all_correct = 0
        cond_counts = {"valid_date": 0, "day": 0, "month": 0, "leap": 0, "decade": 0}

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                batch_size = x.size(0)

                # Forward pass
                if name == "Conditional GAN":
                    noise = torch.randn(batch_size, 64, device=device)
                    out = model(x, noise)
                else:
                    out = model(x)

                # Convert to indices
                if config["type"] == "logits":
                    pred_indices = out.argmax(dim=-1)
                else:
                    pred_indices = out

                for i in range(batch_size):
                    cond_day = tokenizer.days[x[i, 0].item()]
                    cond_month = tokenizer.months[x[i, 1].item()]
                    cond_leap = tokenizer.leaps[x[i, 2].item()]
                    cond_decade = tokenizer.decades[x[i, 3].item()]

                    pred_str = tokenizer.decode_output(pred_indices[i])

                    # Full validation
                    if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                        all_correct += 1

                    # Per-condition breakdown
                    checks = check_individual_conditions(
                        pred_str, cond_day, cond_month, cond_leap, cond_decade,
                    )
                    for key in cond_counts:
                        if checks[key]:
                            cond_counts[key] += 1

                    total += 1

        if total > 0:
            pct = lambda c: f"{(c / total) * 100:.2f}%"
            print(
                f"{name:<20} | {pct(all_correct):>8} | {pct(cond_counts['valid_date']):>8} | "
                f"{pct(cond_counts['day']):>7} | {pct(cond_counts['month']):>8} | "
                f"{pct(cond_counts['leap']):>7} | {pct(cond_counts['decade']):>9} | "
                f"{total:>8}"
            )

    # Print some sample predictions from the best model
    print(f"\n{'=' * 60}")
    print("  Sample Predictions (first 10 from each model)")
    print(f"{'=' * 60}")

    sample_loader = DataLoader(Subset(eval_dataset, list(range(10))), batch_size=10, shuffle=False)
    x_sample, y_sample = next(iter(sample_loader))
    x_sample = x_sample.to(device)

    for name, config in models.items():
        weights_path = config["weights"]
        if not weights_path.exists():
            continue

        model = config["model"]
        model.eval()

        print(f"\n--- {name} ---")
        with torch.no_grad():
            if name == "Conditional GAN":
                noise = torch.randn(x_sample.size(0), 64, device=device)
                out = model(x_sample, noise)
            else:
                out = model(x_sample)

            if config["type"] == "logits":
                pred_indices = out.argmax(dim=-1)
            else:
                pred_indices = out

            for i in range(x_sample.size(0)):
                conds = (
                    f"{tokenizer.days[x_sample[i, 0].item()]} "
                    f"{tokenizer.months[x_sample[i, 1].item()]} "
                    f"{tokenizer.leaps[x_sample[i, 2].item()]} "
                    f"{tokenizer.decades[x_sample[i, 3].item()]}"
                )
                pred = tokenizer.decode_output(pred_indices[i])
                target = tokenizer.decode_output(y_sample[i])
                ok = "OK" if validate_date(
                    pred,
                    tokenizer.days[x_sample[i, 0].item()],
                    tokenizer.months[x_sample[i, 1].item()],
                    tokenizer.leaps[x_sample[i, 2].item()],
                    tokenizer.decades[x_sample[i, 3].item()],
                ) else "FAIL"
                print(f"  {conds} -> {pred} (target: {target}) [{ok}]")


if __name__ == "__main__":
    evaluate_models()