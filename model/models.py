import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=128):
        super(SimpleLSTMGenerator, self).__init__()
        
        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)
        
        self.fc_init_hidden = nn.Linear(emb_dim * 4, hidden_dim)
        self.fc_init_cell = nn.Linear(emb_dim * 4, hidden_dim)
        
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, conditions, target_seq=None):
        day_e = self.day_emb(conditions[:, 0])
        month_e = self.month_emb(conditions[:, 1])
        leap_e = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])
        
        cond_features = torch.cat([day_e, month_e, leap_e, decade_e], dim=1)
        
        h0 = self.fc_init_hidden(cond_features).unsqueeze(0)
        c0 = self.fc_init_cell(cond_features).unsqueeze(0)
        
        if target_seq is not None:
            dec_input = self.char_emb(target_seq[:, :-1])
            output, _ = self.lstm(dec_input, (h0, c0))
            logits = self.fc_out(output)
            return logits
        else:
            batch_size = conditions.size(0)
            device = conditions.device
            
            current_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            h, c = h0, c0
            
            outputs = []
            # Change from 12 to 14 to accommodate the new max_len
            for _ in range(14):
                emb = self.char_emb(current_token)
                out, (h, c) = self.lstm(emb, (h, c))
                logits = self.fc_out(out)
                
                predicted_token = logits.argmax(dim=-1)
                outputs.append(predicted_token)
                
                current_token = predicted_token
                
            return torch.cat(outputs, dim=1)
        
class SmallTransformerGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super(SmallTransformerGenerator, self).__init__()
        
        # Condition Embeddings
        self.day_emb = nn.Embedding(7, d_model)
        self.month_emb = nn.Embedding(12, d_model)
        self.leap_emb = nn.Embedding(2, d_model)
        self.decade_emb = nn.Embedding(41, d_model)
        
        # Sequence Embeddings (Characters + Position)
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(20, d_model) # Max sequence length buffer
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        # Prevents the model from looking ahead at future characters during training
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, conditions, target_seq=None):
        device = conditions.device
        batch_size = conditions.size(0)
        
        # 1. Encode Conditions
        d = self.day_emb(conditions[:, 0]).unsqueeze(1)
        m = self.month_emb(conditions[:, 1]).unsqueeze(1)
        l = self.leap_emb(conditions[:, 2]).unsqueeze(1)
        dec = self.decade_emb(conditions[:, 3]).unsqueeze(1)
        
        memory = torch.cat([d, m, l, dec], dim=1) 
        
        # 2. Decode Sequence
        if target_seq is not None:
            # Shift the sequence for Teacher Forcing (exclude the last token)
            dec_input = target_seq[:, :-1]
            seq_len = dec_input.size(1)
            
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            tgt = self.char_emb(dec_input) + self.pos_emb(positions)
            tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
            
            out = self.transformer(memory, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
            return self.fc_out(out)
        else:
            # Inference
            current_seq = torch.full((batch_size, 1), 1, dtype=torch.long, device=device) 
            
            for _ in range(14): 
                positions = torch.arange(current_seq.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
                tgt = self.char_emb(current_seq) + self.pos_emb(positions)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)
                
                out = self.transformer(memory, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
                
                logits = self.fc_out(out[:, -1, :]) 
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
            return current_seq[:, 1:]
        
        
class Generator(nn.Module):
    def __init__(self, vocab_size, noise_dim=64, emb_dim=16, hidden_dim=256, seq_len=14):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)
        
        input_dim = noise_dim + (emb_dim * 4)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * vocab_size)
        )
        
    def forward(self, conditions, noise):
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        
        cond_features = torch.cat([d, m, l, dec], dim=1)
        x = torch.cat([noise, cond_features], dim=1)
        
        out = self.net(x)
        out = out.view(-1, self.seq_len, self.vocab_size)
        
        # Output probabilities using Gumbel-Softmax (trivial way to make discrete text differentiable)
        return F.gumbel_softmax(out, tau=1.0, hard=True, dim=-1)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=256, seq_len=14):
        super(Discriminator, self).__init__()
        
        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)
        
        # Sequence input size: sequence length * vocabulary size (flattened)
        input_dim = (seq_len * vocab_size) + (emb_dim * 4)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, conditions, seq_probs):
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        
        cond_features = torch.cat([d, m, l, dec], dim=1)
        
        # Flatten the sequence probabilities to feed into the MLP
        seq_flat = seq_probs.view(seq_probs.size(0), -1)
        
        x = torch.cat([seq_flat, cond_features], dim=1)
        return self.net(x)
    
class ConditionalVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, latent_dim=32, hidden_dim=256, seq_len=14):
        super(ConditionalVAE, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.day_emb = nn.Embedding(7, emb_dim)
        self.month_emb = nn.Embedding(12, emb_dim)
        self.leap_emb = nn.Embedding(2, emb_dim)
        self.decade_emb = nn.Embedding(41, emb_dim)

        self.char_emb = nn.Embedding(vocab_size, emb_dim)

        # Encoder
        enc_input_dim = (seq_len * emb_dim) + (emb_dim * 4)
        self.enc_fc1 = nn.Linear(enc_input_dim, hidden_dim)
        self.enc_fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_input_dim = latent_dim + (emb_dim * 4)
        self.dec_fc1 = nn.Linear(dec_input_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, seq_len * vocab_size)

    def encode(self, x, conditions):
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        cond_features = torch.cat([d, m, l, dec], dim=1)

        x_emb = self.char_emb(x).view(x.size(0), -1)
        enc_in = torch.cat([x_emb, cond_features], dim=1)

        h = torch.relu(self.enc_fc1(enc_in))
        mu = self.enc_fc2_mu(h)
        logvar = self.enc_fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, conditions):
        d = self.day_emb(conditions[:, 0])
        m = self.month_emb(conditions[:, 1])
        l = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        cond_features = torch.cat([d, m, l, dec], dim=1)

        dec_in = torch.cat([z, cond_features], dim=1)
        h = torch.relu(self.dec_fc1(dec_in))
        out = self.dec_fc2(h)
        return out.view(-1, self.seq_len, self.vocab_size)

    def forward(self, conditions, x=None):
        if x is not None:
            # Training phase
            mu, logvar = self.encode(x, conditions)
            z = self.reparameterize(mu, logvar)
            out = self.decode(z, conditions)
            return out, mu, logvar
        else:
            # Inference phase
            z = torch.randn(conditions.size(0), self.latent_dim, device=conditions.device)
            out = self.decode(z, conditions)
            return out