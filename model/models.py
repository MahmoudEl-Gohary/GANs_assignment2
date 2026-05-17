"""Neural network architectures for conditional date generation.

Contains four models:
1. SimpleLSTMGenerator -- Autoregressive LSTM decoder (course model).
2. SmallTransformerGenerator -- Encoder-decoder Transformer (course model).
3. Generator / Discriminator -- Conditional GAN (required).
4. ConditionalVAE -- Conditional Variational Autoencoder (extra).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTMGenerator(nn.Module):
    """Autoregressive LSTM that generates a date character-by-character.

    Conditions are embedded and:
    1. Projected to initialize the LSTM hidden/cell states.
    2. Concatenated to the character embedding at EVERY decoder step.

    This dual injection ensures the model retains access to all conditions
    (especially day-of-week) even when generating the final date digits.

    Args:
        vocab_size: Number of output characters (including PAD/SOS/EOS).
        emb_dim: Embedding dimension for both conditions and characters.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (only active if num_layers > 1).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Condition embeddings
        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)

        # Project concatenated conditions to per-layer LSTM init states
        cond_dim = emb_dim * 4
        self.fc_init_hidden = nn.Linear(cond_dim, hidden_dim * num_layers)
        self.fc_init_cell = nn.Linear(cond_dim, hidden_dim * num_layers)

        # Character-level decoder -- input is char_emb + condition_features
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim + cond_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def _embed_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all four conditions into a flat vector."""
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        return torch.cat([d, m, l, dec], dim=1)

    def _init_hidden(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build initial (h0, c0) from condition embeddings."""
        batch_size = cond.size(0)
        h0 = self.fc_init_hidden(cond).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = self.fc_init_cell(cond).view(batch_size, self.num_layers, self.hidden_dim)
        # LSTM expects (num_layers, batch, hidden)
        return h0.permute(1, 0, 2).contiguous(), c0.permute(1, 0, 2).contiguous()

    def forward(
        self,
        conditions: torch.Tensor,
        target_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            conditions: (batch, 4) condition indices.
            target_seq: (batch, seq_len) target token indices for teacher forcing.
                        If None, runs autoregressive inference.

        Returns:
            Training: logits of shape (batch, seq_len-1, vocab_size).
            Inference: predicted indices of shape (batch, 12).
        """
        cond_features = self._embed_conditions(conditions)
        h0, c0 = self._init_hidden(cond_features)

        if target_seq is not None:
            char_embs = self.char_emb(target_seq[:, :-1])  # (batch, seq-1, emb)
            seq_len = char_embs.size(1)
            # Broadcast conditions to every time step
            cond_expanded = cond_features.unsqueeze(1).expand(-1, seq_len, -1)
            dec_input = torch.cat([char_embs, cond_expanded], dim=2)

            output, _ = self.lstm(dec_input, (h0, c0))
            logits = self.fc_out(output)
            return logits
        else:
            batch_size = conditions.size(0)
            device = conditions.device

            current_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            h, c = h0, c0

            outputs: list[torch.Tensor] = []
            for _ in range(12):
                char_e = self.char_emb(current_token)  # (batch, 1, emb)
                cond_e = cond_features.unsqueeze(1)     # (batch, 1, cond_dim)
                lstm_input = torch.cat([char_e, cond_e], dim=2)

                out, (h, c) = self.lstm(lstm_input, (h, c))
                logits = self.fc_out(out)

                predicted_token = logits.argmax(dim=-1)
                outputs.append(predicted_token)

                current_token = predicted_token

            return torch.cat(outputs, dim=1)


class SmallTransformerGenerator(nn.Module):
    """Encoder-decoder Transformer for conditional date generation.

    The encoder treats the 4 condition embeddings as a length-4 source sequence
    for cross-attention. Additionally, a projected condition vector is added
    to every decoder position, ensuring persistent condition access.

    Args:
        vocab_size: Number of output characters.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder/decoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Condition Embeddings (for cross-attention memory)
        self.day_emb = nn.Embedding(7, d_model)
        self.month_emb = nn.Embedding(12, d_model)
        self.leap_emb = nn.Embedding(2, d_model)
        self.decade_emb = nn.Embedding(41, d_model)

        # Condition projection for direct decoder injection
        cond_dim = d_model * 4
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # Sequence Embeddings (Characters + Position)
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(20, d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Embed conditions as a (batch, 4, d_model) memory sequence."""
        d = self.day_emb(conditions[:, 0]).unsqueeze(1)
        m = self.month_emb(conditions[:, 1]).unsqueeze(1)
        l = self.leap_emb(conditions[:, 2]).unsqueeze(1)
        dec = self.decade_emb(conditions[:, 3]).unsqueeze(1)
        return torch.cat([d, m, l, dec], dim=1)

    def _get_cond_bias(self, conditions: torch.Tensor) -> torch.Tensor:
        """Project flat condition vector for additive injection into decoder.

        Returns:
            (batch, 1, d_model) bias vector.
        """
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        flat = torch.cat([d, m, l, dec], dim=1)
        return self.cond_proj(flat).unsqueeze(1)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate an upper-triangular causal mask for the decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float('-inf'))

    def forward(
        self,
        conditions: torch.Tensor,
        target_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            conditions: (batch, 4) condition indices.
            target_seq: (batch, seq_len) target token indices for teacher forcing.

        Returns:
            Training: logits of shape (batch, seq_len-1, vocab_size).
            Inference: predicted indices of shape (batch, 12).
        """
        device = conditions.device
        batch_size = conditions.size(0)

        memory = self._encode_conditions(conditions)
        cond_bias = self._get_cond_bias(conditions)  # (batch, 1, d_model)

        if target_seq is not None:
            # Teacher forcing
            dec_input = target_seq[:, :-1]
            seq_len = dec_input.size(1)

            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            tgt = self.char_emb(dec_input) + self.pos_emb(positions) + cond_bias
            tgt_mask = self._generate_causal_mask(seq_len, device)

            out = self.transformer(memory, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
            return self.fc_out(out)
        else:
            # Autoregressive inference
            current_seq = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)

            for _ in range(12):
                positions = torch.arange(current_seq.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
                tgt = self.char_emb(current_seq) + self.pos_emb(positions) + cond_bias
                tgt_mask = self._generate_causal_mask(tgt.size(1), device)

                out = self.transformer(memory, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)

                logits = self.fc_out(out[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)

                current_seq = torch.cat([current_seq, next_token], dim=1)

            return current_seq[:, 1:]


class Generator(nn.Module):
    """Generator for the conditional GAN.

    Takes random noise + condition embeddings and produces a sequence of
    soft one-hot vectors via Gumbel-Softmax.

    Args:
        vocab_size: Number of output characters.
        noise_dim: Dimension of the input noise vector.
        emb_dim: Condition embedding dimension.
        hidden_dim: MLP hidden dimension.
        seq_len: Length of the output sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        noise_dim: int = 64,
        emb_dim: int = 32,
        hidden_dim: int = 512,
        seq_len: int = 12,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)

        input_dim = noise_dim + (emb_dim * 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * vocab_size),
        )

    def forward(self, conditions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Generate a sequence of soft one-hot vectors.

        Args:
            conditions: (batch, 4) condition indices.
            noise: (batch, noise_dim) random noise.

        Returns:
            (batch, seq_len, vocab_size) soft one-hot via Gumbel-Softmax.
        """
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])

        cond_features = torch.cat([d, m, l, dec], dim=1)
        x = torch.cat([noise, cond_features], dim=1)

        out = self.net(x)
        out = out.view(-1, self.seq_len, self.vocab_size)

        return F.gumbel_softmax(out, tau=0.5, hard=True, dim=-1)


class Discriminator(nn.Module):
    """Discriminator for the conditional GAN.

    Takes condition embeddings + flattened sequence probabilities and outputs
    a real/fake probability.

    Args:
        vocab_size: Number of output characters.
        emb_dim: Condition embedding dimension.
        hidden_dim: MLP hidden dimension.
        seq_len: Length of the input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        hidden_dim: int = 512,
        seq_len: int = 12,
    ) -> None:
        super().__init__()

        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)

        input_dim = (seq_len * vocab_size) + (emb_dim * 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, conditions: torch.Tensor, seq_probs: torch.Tensor) -> torch.Tensor:
        """Classify a sequence as real or fake.

        Args:
            conditions: (batch, 4) condition indices.
            seq_probs: (batch, seq_len, vocab_size) one-hot or soft probabilities.

        Returns:
            (batch, 1) real/fake probability.
        """
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])

        cond_features = torch.cat([d, m, l, dec], dim=1)

        seq_flat = seq_probs.view(seq_probs.size(0), -1)

        x = torch.cat([seq_flat, cond_features], dim=1)
        return self.net(x)


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for date generation.

    Encodes a target date + conditions into a latent space, then decodes
    from a latent sample + conditions back into a date sequence.

    Args:
        vocab_size: Number of output characters.
        emb_dim: Embedding dimension.
        latent_dim: Latent space dimension.
        hidden_dim: MLP hidden dimension.
        seq_len: Length of the output sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        seq_len: int = 12,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)

        self.char_emb = nn.Embedding(vocab_size, emb_dim)

        cond_dim = emb_dim * 4

        # Encoder
        enc_input_dim = (seq_len * emb_dim) + cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_input_dim = latent_dim + cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * vocab_size),
        )

    def _embed_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all four conditions."""
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        return torch.cat([d, m, l, dec], dim=1)

    def encode(
        self, x: torch.Tensor, conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a target sequence + conditions into (mu, logvar).

        Args:
            x: (batch, seq_len) target token indices.
            conditions: (batch, 4) condition indices.

        Returns:
            Tuple of (mu, logvar), each (batch, latent_dim).
        """
        cond_features = self._embed_conditions(conditions)
        x_emb = self.char_emb(x).view(x.size(0), -1)
        enc_in = torch.cat([x_emb, cond_features], dim=1)

        h = self.encoder(enc_in)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample + conditions into logits.

        Args:
            z: (batch, latent_dim) latent sample.
            conditions: (batch, 4) condition indices.

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        cond_features = self._embed_conditions(conditions)
        dec_in = torch.cat([z, cond_features], dim=1)
        out = self.decoder(dec_in)
        return out.view(-1, self.seq_len, self.vocab_size)

    def forward(
        self,
        conditions: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            conditions: (batch, 4) condition indices.
            x: (batch, seq_len) target token indices. If None, inference mode.

        Returns:
            Training: (logits, mu, logvar).
            Inference: logits only.
        """
        if x is not None:
            mu, logvar = self.encode(x, conditions)
            z = self.reparameterize(mu, logvar)
            out = self.decode(z, conditions)
            return out, mu, logvar
        else:
            z = torch.randn(conditions.size(0), self.latent_dim, device=conditions.device)
            out = self.decode(z, conditions)
            return out