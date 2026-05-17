import torch
from torch.utils.data import Dataset

class DatesTokenizer:
    def __init__(self):
        self.days = ['[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]']
        self.months = ['[JAN]', '[FEB]', '[MAR]', '[APR]', '[MAY]', '[JUN]', 
                       '[JUL]', '[AUG]', '[SEP]', '[OCT]', '[NOV]', '[DEC]']
        self.leaps = ['[False]', '[True]']
        self.decades = [f"[{i}]" for i in range(180, 221)]
        
        self.chars = ['<PAD>', '<SOS>', '<EOS>', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
    def encode_input(self, day, month, leap, decade):
        day_idx = self.days.index(day)
        month_idx = self.months.index(month)
        leap_idx = self.leaps.index(leap)
        decade_idx = self.decades.index(decade)
        return torch.tensor([day_idx, month_idx, leap_idx, decade_idx], dtype=torch.long)
        
    def encode_output(self, date_str, max_len=14):
        # Reverse format from dd-mm-yyyy to yyyy-mm-dd
        d, m, y = date_str.split('-')
        formatted_date = f"{y}-{m}-{d}"
        
        tokens = ['<SOS>'] + list(formatted_date) + ['<EOS>']
        
        while len(tokens) < max_len:
            tokens.append('<PAD>')
            
        indices = [self.char_to_idx[c] for c in tokens]
        return torch.tensor(indices, dtype=torch.long)
        
    def decode_output(self, indices):
        chars = []
        for idx in indices:
            val = idx.item() if isinstance(idx, torch.Tensor) else idx
            char = self.idx_to_char[val]
            
            if char == '<EOS>':
                break
            if char not in ['<PAD>', '<SOS>']:
                chars.append(char)
                
        decoded_str = "".join(chars)
        
        # Revert back to dd-mm-yyyy for validation and final output
        parts = decoded_str.split('-')
        if len(parts) == 3:
            y, m, d = parts
            return f"{d}-{m}-{y}"
        return decoded_str
class DatesDataset(Dataset):
    def __init__(self, file_path, tokenizer, is_inference=False):
        self.tokenizer = tokenizer
        self.is_inference = is_inference
        self.samples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                if self.is_inference:
                    self.samples.append(parts[:4])
                else:
                    self.samples.append((parts[:4], parts[4]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.is_inference:
            conditions = self.samples[idx]
            x = self.tokenizer.encode_input(*conditions)
            return x
        else:
            conditions, date_str = self.samples[idx]
            x = self.tokenizer.encode_input(*conditions)
            y = self.tokenizer.encode_output(date_str)
            return x, y