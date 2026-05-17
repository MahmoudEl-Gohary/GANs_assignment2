"""Generate dates from an input file of conditions.

Usage:
    python predict.py -i data/example_input.txt -o data/example_output.txt
    python predict.py -i data/example_input.txt -o data/example_output.txt --model lstm
"""

import argparse
from pathlib import Path

import torch

from model.dataset import DatesTokenizer
from model.models import (
    SimpleLSTMGenerator,
    SmallTransformerGenerator,
    Generator,
    ConditionalVAE,
)

WEIGHTS_DIR = Path("model/weights")

# Registry mapping model name to (class, weights filename, output_type)
MODEL_REGISTRY: dict[str, tuple[type, str, str]] = {
    "lstm": (SimpleLSTMGenerator, "lstm_weights.pth", "indices"),
    "transformer": (SmallTransformerGenerator, "transformer_weights.pth", "indices"),
    "gan": (Generator, "gan_generator_weights.pth", "logits"),
    "cvae": (ConditionalVAE, "cvae_weights.pth", "logits"),
}


def predict(input_path: str, output_path: str, model_name: str = "lstm") -> None:
    """Generate dates for each line of conditions in the input file.

    Args:
        input_path: Path to the input text file (conditions only).
        output_path: Path to write the output file (conditions + date).
        model_name: Which model to use for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DatesTokenizer()

    # Load selected model
    model_cls, weights_file, output_type = MODEL_REGISTRY[model_name]
    weights_path = WEIGHTS_DIR / weights_file

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    model = model_cls(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts or len(parts) < 4:
                continue

            conditions = parts[:4]
            x = tokenizer.encode_input(*conditions).unsqueeze(0).to(device)

            with torch.no_grad():
                if model_name == "gan":
                    noise = torch.randn(1, 64, device=device)
                    out = model(x, noise)
                else:
                    out = model(x)

                if output_type == "logits":
                    pred_indices = out.argmax(dim=-1)[0]
                else:
                    pred_indices = out[0]

                date_str = tokenizer.decode_output(pred_indices)

            conditions_str = " ".join(conditions)
            fout.write(f"{conditions_str} {date_str}\n")

    print(f"Predictions written to {output_path} (model: {model_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dates based on calendar conditions.",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input text file.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output text file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default="lstm",
        help="Model to use for inference (default: lstm).",
    )

    args = parser.parse_args()
    predict(args.input, args.output, args.model)