import os
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime

from model.dataset import DatesTokenizer, DatesDataset
from model.models import SimpleLSTMGenerator, SmallTransformerGenerator, Generator, ConditionalVAE

def validate_date(date_str, cond_day, cond_month, cond_leap, cond_decade):
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        
        # Check Leap Year
        year = dt.year
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if is_leap != (cond_leap == "[True]"): return False
            
        # Check Decade
        expected_decade_start = int(cond_decade[1:-1]) * 10
        if not (expected_decade_start <= year < expected_decade_start + 10): return False
            
        # Check Month
        months_map = {'[JAN]': 1, '[FEB]': 2, '[MAR]': 3, '[APR]': 4, '[MAY]': 5, '[JUN]': 6,
                      '[JUL]': 7, '[AUG]': 8, '[SEP]': 9, '[OCT]': 10, '[NOV]': 11, '[DEC]': 12}
        if dt.month != months_map[cond_month]: return False
            
        # Check Day
        days_map = {'[MON]': 0, '[TUE]': 1, '[WED]': 2, '[THU]': 3, '[FRI]': 4, '[SAT]': 5, '[SUN]': 6}
        if dt.weekday() != days_map[cond_day]: return False
            
        return True
    except ValueError:
        return False

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}\n")
    
    tokenizer = DatesTokenizer()
    dataset = DatesDataset("data/data.txt", tokenizer)
    vocab_size = len(tokenizer.chars)
    
    # Evaluate on a fixed random subset of 5,000 samples for speed (Bonus: using manual seed)
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))[:10000].tolist()
    eval_dataset = Subset(dataset, indices)
    dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    # 1. Initialize all models
    models = {
        "Simple LSTM": {
            "model": SimpleLSTMGenerator(vocab_size=vocab_size, hidden_dim=256).to(device),
            "weights": "model/weights/lstm_weights.pth",
            "type": "indices"
        },
        "Small Transformer": {
            "model": SmallTransformerGenerator(vocab_size=vocab_size).to(device),
            "weights": "model/weights/transformer_weights.pth",
            "type": "indices"
        },
        "Conditional GAN": {
            "model": Generator(vocab_size=vocab_size).to(device),
            "weights": "model/weights/gan_generator_weights.pth",
            "type": "logits"
        },
        "Conditional VAE": {
            "model": ConditionalVAE(vocab_size=vocab_size).to(device),
            "weights": "model/weights/cvae_weights.pth",
            "type": "logits"
        }
    }

    # 2. Run Evaluation
    print(f"{'Model Name':<20} | {'Valid Dates (%)':<15} | {'Status'}")
    print("-" * 60)

    for name, config in models.items():
        model = config["model"]
        weights_path = config["weights"]
        
        if not os.path.exists(weights_path):
            print(f"{name:<20} | {'N/A':<15} | Missing weights file")
            continue
            
        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                batch_size = x.size(0)
                
                # Forward pass handling based on model architecture
                if name == "Conditional GAN":
                    noise = torch.randn(batch_size, 64, device=device)
                    out = model(x, noise)
                else:
                    out = model(x)
                
                # Convert logits to indices if necessary
                if config["type"] == "logits":
                    pred_indices = out.argmax(dim=-1)
                else:
                    pred_indices = out
                
                # Evaluate each sequence in the batch
                for i in range(batch_size):
                    cond_day = tokenizer.days[x[i, 0].item()]
                    cond_month = tokenizer.months[x[i, 1].item()]
                    cond_leap = tokenizer.leaps[x[i, 2].item()]
                    cond_decade = tokenizer.decades[x[i, 3].item()]
                    
                    pred_str = tokenizer.decode_output(pred_indices[i])
                    
                    if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                        correct += 1
                    total += 1
                    
        accuracy = (correct / total) * 100
        print(f"{name:<20} | {accuracy:>14.2f}% | Evaluated {total} samples")

if __name__ == "__main__":
    evaluate_models()