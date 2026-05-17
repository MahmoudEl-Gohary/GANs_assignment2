"""Dataset and tokenizer for the conditional date generation task."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class DatesTokenizer:
    """Encodes/decodes conditions and date strings for the date generation models.

    The output date is reordered from dd-mm-yyyy to yyyy-mm-dd so that the
    model predicts the most constrained digits (year/decade) first.
    """

    def __init__(self) -> None:
        self.days: list[str] = [
            '[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]',
        ]
        self.months: list[str] = [
            '[JAN]', '[FEB]', '[MAR]', '[APR]', '[MAY]', '[JUN]',
            '[JUL]', '[AUG]', '[SEP]', '[OCT]', '[NOV]', '[DEC]',
        ]
        self.leaps: list[str] = ['[False]', '[True]']
        self.decades: list[str] = [f"[{i}]" for i in range(180, 221)]

        self.chars: list[str] = [
            '<PAD>', '<SOS>', '<EOS>',
            '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        ]
        self.char_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

        # Convenience constants
        self.pad_idx: int = self.char_to_idx['<PAD>']
        self.sos_idx: int = self.char_to_idx['<SOS>']
        self.eos_idx: int = self.char_to_idx['<EOS>']
        self.vocab_size: int = len(self.chars)

    def encode_input(self, day: str, month: str, leap: str, decade: str) -> torch.Tensor:
        """Encode the four condition tokens into a tensor of indices.

        Args:
            day: Day-of-week token, e.g. '[MON]'.
            month: Month token, e.g. '[JAN]'.
            leap: Leap year token, '[True]' or '[False]'.
            decade: Decade token, e.g. '[192]'.

        Returns:
            LongTensor of shape (4,).
        """
        day_idx = self.days.index(day)
        month_idx = self.months.index(month)
        leap_idx = self.leaps.index(leap)
        decade_idx = self.decades.index(decade)
        return torch.tensor([day_idx, month_idx, leap_idx, decade_idx], dtype=torch.long)

    def encode_output(self, date_str: str, max_len: int = 12) -> torch.Tensor:
        """Encode a date string (dd-mm-yyyy) into a padded index tensor.

        The date is internally reordered to yyyy-mm-dd with zero-padded
        day/month so that every sequence is exactly the same length.
        This eliminates positional ambiguity for the model.

        Args:
            date_str: Date in dd-mm-yyyy format (day/month may lack leading zeros).
            max_len: Maximum token sequence length including SOS/EOS.

        Returns:
            LongTensor of shape (max_len,).
        """
        d, m, y = date_str.split('-')
        # Zero-pad day and month to 2 digits for fixed-length output
        formatted_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        tokens = ['<SOS>'] + list(formatted_date) + ['<EOS>']

        while len(tokens) < max_len:
            tokens.append('<PAD>')

        indices = [self.char_to_idx[c] for c in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def decode_output(self, indices: torch.Tensor) -> str:
        """Decode a tensor of character indices back into a dd-mm-yyyy string.

        Strips leading zeros from day/month to match the original data format.

        Args:
            indices: 1-D tensor of character indices.

        Returns:
            Date string in dd-mm-yyyy format (no leading zeros on day/month).
        """
        chars: list[str] = []
        for idx in indices:
            val = idx.item() if isinstance(idx, torch.Tensor) else idx
            char = self.idx_to_char[val]

            if char == '<EOS>':
                break
            if char not in ('<PAD>', '<SOS>'):
                chars.append(char)

        decoded_str = "".join(chars)

        # Revert back to dd-mm-yyyy for validation and final output
        parts = decoded_str.split('-')
        if len(parts) == 3:
            y, m, d = parts
            # Strip leading zeros to match original format (e.g., 01 -> 1)
            d_stripped = str(int(d)) if d.isdigit() else d
            m_stripped = str(int(m)) if m.isdigit() else m
            return f"{d_stripped}-{m_stripped}-{y}"
        return decoded_str


class DatesDataset(Dataset):
    """PyTorch dataset for the dates generation task.

    Each sample is either (conditions_tensor, date_tensor) for training,
    or just conditions_tensor for inference.

    Args:
        file_path: Path to the data text file.
        tokenizer: DatesTokenizer instance.
        is_inference: If True, expect input-only lines (no output date).
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: DatesTokenizer,
        is_inference: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.is_inference = is_inference
        self.samples: list = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if self.is_inference:
                    self.samples.append(parts[:4])
                else:
                    self.samples.append((parts[:4], parts[4]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...] | torch.Tensor:
        if self.is_inference:
            conditions = self.samples[idx]
            x = self.tokenizer.encode_input(*conditions)
            return x
        else:
            conditions, date_str = self.samples[idx]
            x = self.tokenizer.encode_input(*conditions)
            y = self.tokenizer.encode_output(date_str)
            return x, y