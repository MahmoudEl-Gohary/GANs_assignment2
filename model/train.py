"""Training scripts for all four date generation models.

Usage:
    python -m model.train --model lstm
    python -m model.train --model transformer
    python -m model.train --model gan
    python -m model.train --model cvae
    python -m model.train --model all
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.dataset import DatesTokenizer, DatesDataset
from model.models import (
    SimpleLSTMGenerator,
    SmallTransformerGenerator,
    Generator,
    Discriminator,
    ConditionalVAE,
)
from model.utils import validate_date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DATA_PATH = Path("data/data.txt")
WEIGHTS_DIR = Path("model/weights")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup_device() -> torch.device:
    """Select CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    return device


def _seed_everything(seed: int = SEED) -> None:
    """Set manual seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_data_splits(
    dataset: DatesDataset,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset 90/10 and return train/test DataLoaders."""
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: DatesTokenizer,
    device: torch.device,
) -> float:
    """Compute calendar validation accuracy on a dataloader.

    Args:
        model: Trained model in eval mode.
        dataloader: DataLoader yielding (conditions, targets).
        tokenizer: DatesTokenizer for decoding.
        device: torch device.

    Returns:
        Fraction of generated dates passing all four conditions.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)

            predictions = model(x)

            for i in range(batch_size):
                cond_day = tokenizer.days[x[i, 0].item()]
                cond_month = tokenizer.months[x[i, 1].item()]
                cond_leap = tokenizer.leaps[x[i, 2].item()]
                cond_decade = tokenizer.decades[x[i, 3].item()]

                pred_str = tokenizer.decode_output(predictions[i])

                if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Model-specific trainers
# ---------------------------------------------------------------------------

def train_lstm(epochs: int = 30) -> None:
    """Train the SimpleLSTMGenerator."""
    _seed_everything()
    device = _setup_device()

    tokenizer = DatesTokenizer()
    dataset = DatesDataset(DATA_PATH, tokenizer)
    train_loader, test_loader = _get_data_splits(dataset)

    model = SimpleLSTMGenerator(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"LSTM Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, target_seq=y)

            logits = logits.reshape(-1, tokenizer.vocab_size)
            targets = y[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        val_acc = evaluate_model(model, test_loader, tokenizer, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best weights
        if val_acc > best_acc:
            best_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_DIR / "lstm_weights.pth")
            print(f"  -> New best! Saved weights (acc={best_acc:.4f})")

    print(f"LSTM training complete. Best accuracy: {best_acc:.4f}")


def train_transformer(epochs: int = 30) -> None:
    """Train the SmallTransformerGenerator."""
    _seed_everything()
    device = _setup_device()

    tokenizer = DatesTokenizer()
    dataset = DatesDataset(DATA_PATH, tokenizer)
    train_loader, test_loader = _get_data_splits(dataset)

    model = SmallTransformerGenerator(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Transformer Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, target_seq=y)

            logits = logits.reshape(-1, tokenizer.vocab_size)
            targets = y[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        val_acc = evaluate_model(model, test_loader, tokenizer, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_DIR / "transformer_weights.pth")
            print(f"  -> New best! Saved weights (acc={best_acc:.4f})")

    print(f"Transformer training complete. Best accuracy: {best_acc:.4f}")


def train_gan(epochs: int = 30) -> None:
    """Train the conditional GAN (Generator + Discriminator)."""
    _seed_everything()
    device = _setup_device()

    tokenizer = DatesTokenizer()
    dataset = DatesDataset(DATA_PATH, tokenizer)
    train_loader, test_loader = _get_data_splits(dataset, batch_size=64)

    vocab_size = tokenizer.vocab_size
    noise_dim = 64

    generator = Generator(vocab_size=vocab_size).to(device)
    discriminator = Discriminator(vocab_size=vocab_size).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"GAN Epoch {epoch + 1}/{epochs}", leave=False)):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            real_labels = torch.ones(batch_size, 1, device=device) * 0.9  # label smoothing
            fake_labels = torch.zeros(batch_size, 1, device=device) + 0.1

            real_seq_one_hot = F.one_hot(y, num_classes=vocab_size).float()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            real_preds = discriminator(x, real_seq_one_hot)
            d_loss_real = criterion(real_preds, real_labels)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_seq_probs = generator(x, noise)
            fake_preds = discriminator(x, fake_seq_probs.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_seq_probs = generator(x, noise)
            fake_preds_for_g = discriminator(x, fake_seq_probs)
            g_loss = criterion(fake_preds_for_g, torch.ones(batch_size, 1, device=device))

            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g = total_g_loss / len(train_loader)
        avg_d = total_d_loss / len(train_loader)

        # Evaluate GAN generator
        generator.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_eval, _ in test_loader:
                x_eval = x_eval.to(device)
                bs = x_eval.size(0)
                noise = torch.randn(bs, noise_dim, device=device)
                out = generator(x_eval, noise)
                pred_indices = out.argmax(dim=-1)

                for j in range(bs):
                    cond_day = tokenizer.days[x_eval[j, 0].item()]
                    cond_month = tokenizer.months[x_eval[j, 1].item()]
                    cond_leap = tokenizer.leaps[x_eval[j, 2].item()]
                    cond_decade = tokenizer.decades[x_eval[j, 3].item()]

                    pred_str = tokenizer.decode_output(pred_indices[j])
                    if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                        correct += 1
                    total += 1

        val_acc = correct / total if total > 0 else 0.0
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), WEIGHTS_DIR / "gan_generator_weights.pth")
    torch.save(discriminator.state_dict(), WEIGHTS_DIR / "gan_discriminator_weights.pth")
    print("GAN training complete. Weights saved.")


def _cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    kl_weight: float = 0.01,
) -> torch.Tensor:
    """CVAE loss = reconstruction + KL divergence.

    Args:
        recon_x: (batch, seq_len, vocab_size) reconstructed logits.
        x: (batch, seq_len) target indices.
        mu: (batch, latent_dim) encoder mean.
        logvar: (batch, latent_dim) encoder log-variance.
        criterion: CrossEntropyLoss instance.
        kl_weight: Weight for the KL divergence term.

    Returns:
        Scalar loss.
    """
    recon_loss = criterion(recon_x.view(-1, recon_x.size(-1)), x.view(-1))

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0)

    return recon_loss + (kl_weight * kld)


def train_cvae(epochs: int = 30) -> None:
    """Train the Conditional VAE."""
    _seed_everything()
    device = _setup_device()

    tokenizer = DatesTokenizer()
    dataset = DatesDataset(DATA_PATH, tokenizer)
    train_loader, test_loader = _get_data_splits(dataset)

    model = ConditionalVAE(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"CVAE Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x, y)
            loss = _cvae_loss(recon_batch, y, mu, logvar, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Evaluate CVAE -- use logits -> argmax at inference
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_eval, _ in test_loader:
                x_eval = x_eval.to(device)
                bs = x_eval.size(0)
                out = model(x_eval)  # inference mode, no target
                pred_indices = out.argmax(dim=-1)

                for j in range(bs):
                    cond_day = tokenizer.days[x_eval[j, 0].item()]
                    cond_month = tokenizer.months[x_eval[j, 1].item()]
                    cond_leap = tokenizer.leaps[x_eval[j, 2].item()]
                    cond_decade = tokenizer.decades[x_eval[j, 3].item()]

                    pred_str = tokenizer.decode_output(pred_indices[j])
                    if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                        correct += 1
                    total += 1

        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_DIR / "cvae_weights.pth")
            print(f"  -> New best! Saved weights (acc={best_acc:.4f})")

    print(f"CVAE training complete. Best accuracy: {best_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TRAINERS = {
    "lstm": train_lstm,
    "transformer": train_transformer,
    "gan": train_gan,
    "cvae": train_cvae,
}


def main() -> None:
    """Parse CLI arguments and dispatch to the selected trainer."""
    parser = argparse.ArgumentParser(description="Train date generation models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(TRAINERS.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name, trainer in TRAINERS.items():
            print(f"\n{'=' * 60}")
            print(f"  Training: {name.upper()}")
            print(f"{'=' * 60}\n")
            trainer(epochs=args.epochs)
    else:
        TRAINERS[args.model](epochs=args.epochs)


if __name__ == "__main__":
    main()