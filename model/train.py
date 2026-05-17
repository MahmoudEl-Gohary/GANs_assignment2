import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datetime import datetime

from model.dataset import DatesTokenizer, DatesDataset
from model.models import SimpleLSTMGenerator, SmallTransformerGenerator, Generator, Discriminator, ConditionalVAE

# Custom Evaluation Metric: Calendar Validation
def validate_date(date_str, cond_day, cond_month, cond_leap, cond_decade):
    try:
        # 1. Parse string to datetime
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        
        # 2. Check Leap Year
        year = dt.year
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        expected_leap = cond_leap == "[True]"
        if is_leap != expected_leap:
            return False
            
        # 3. Check Decade
        expected_decade_start = int(cond_decade[1:-1]) * 10
        if not (expected_decade_start <= year < expected_decade_start + 10):
            return False
            
        # 4. Check Month
        months_map = {'[JAN]': 1, '[FEB]': 2, '[MAR]': 3, '[APR]': 4, '[MAY]': 5, '[JUN]': 6,
                      '[JUL]': 7, '[AUG]': 8, '[SEP]': 9, '[OCT]': 10, '[NOV]': 11, '[DEC]': 12}
        if dt.month != months_map[cond_month]:
            return False
            
        # 5. Check Day
        days_map = {'[MON]': 0, '[TUE]': 1, '[WED]': 2, '[THU]': 3, '[FRI]': 4, '[SAT]': 5, '[SUN]': 6}
        if dt.weekday() != days_map[cond_day]:
            return False
            
        return True
    except ValueError:
        return False # Invalid calendar date (e.g. 30-2-2020)

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)
            
            # Generate sequences
            predictions = model(x)
            
            for i in range(batch_size):
                # Decode conditions
                cond_day = tokenizer.days[x[i, 0].item()]
                cond_month = tokenizer.months[x[i, 1].item()]
                cond_leap = tokenizer.leaps[x[i, 2].item()]
                cond_decade = tokenizer.decades[x[i, 3].item()]
                
                # Decode prediction
                pred_str = tokenizer.decode_output(predictions[i])
                
                if validate_date(pred_str, cond_day, cond_month, cond_leap, cond_decade):
                    correct += 1
                total += 1
                
    return correct / total

def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Setup Data
    tokenizer = DatesTokenizer()
    dataset = DatesDataset("data/data.txt", tokenizer)
    
    # Split data (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Setup Model - LSTM (Update hidden_dim)
    model = SimpleLSTMGenerator(vocab_size=len(tokenizer.chars), hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
    
    # Increase epochs
    epochs = 15
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # y[:, :-1] is the input to the decoder, y[:, 1:] is the target to predict
            logits = model(x, target_seq=y)
            
            # Flatten for CrossEntropy
            logits = logits.view(-1, len(tokenizer.chars))
            targets = y[:, 1:].contiguous().view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 4. Evaluate using custom metric
        val_acc = evaluate_model(model, test_loader, tokenizer, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    # Save weights - LSTM
    torch.save(model.state_dict(), "model/weights/lstm_weights.pth")
    
    print("Training complete. Weights saved.")
    
    
def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Setup Data
    tokenizer = DatesTokenizer()
    dataset = DatesDataset("data/data.txt", tokenizer)
    
    # Split data (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # # 2. Setup Model - LSTM (Update hidden_dim)
    # model = SimpleLSTMGenerator(vocab_size=len(tokenizer.chars), hidden_dim=256).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 2.  Setup Model - Transformer
    model = SmallTransformerGenerator(vocab_size=len(tokenizer.chars)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
    
    # Increase epochs
    epochs = 15
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # y[:, :-1] is the input to the decoder, y[:, 1:] is the target to predict
            logits = model(x, target_seq=y)
            
            # Flatten for CrossEntropy
            logits = logits.view(-1, len(tokenizer.chars))
            targets = y[:, 1:].contiguous().view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 4. Evaluate using custom metric
        val_acc = evaluate_model(model, test_loader, tokenizer, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    # Save weights - LSTM
    # torch.save(model.state_dict(), "model/weights/lstm_weights.pth")
    
    # Save weights - Transformerr
    torch.save(model.state_dict(), "model/weights/transformer_weights.pth")
    print("Training complete. Weights saved.")

def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GAN on device: {device}")
    
    tokenizer = DatesTokenizer()
    dataset = DatesDataset("data/data.txt", tokenizer)
    vocab_size = len(tokenizer.chars)
    
    # Smaller batches to stabilize GAN training
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    generator = Generator(vocab_size=vocab_size).to(device)
    discriminator = Discriminator(vocab_size=vocab_size).to(device)
    
    # Standard GAN optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    epochs = 15
    noise_dim = 64
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            # Ground truth labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Convert real sequence to one-hot to match generator output shape
            real_seq_one_hot = F.one_hot(y, num_classes=vocab_size).float()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real Loss
            real_preds = discriminator(x, real_seq_one_hot)
            d_loss_real = criterion(real_preds, real_labels)
            
            # Fake Loss
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_seq_probs = generator(x, noise)
            fake_preds = discriminator(x, fake_seq_probs.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)
            
            # Update D
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake dates and see if D thinks they are real
            fake_preds_for_g = discriminator(x, fake_seq_probs)
            g_loss = criterion(fake_preds_for_g, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            # Print sample to monitor visually (every 1000 batches)
            if i % 1000 == 0:
                with torch.no_grad():
                    sample_probs = fake_seq_probs[0]
                    sample_indices = sample_probs.argmax(dim=-1)
                    decoded_str = tokenizer.decode_output(sample_indices)
                    print(f"Batch {i} - Generated Sample: {decoded_str}")
            
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {total_d_loss/len(dataloader):.4f} | G Loss: {total_g_loss/len(dataloader):.4f}")

    os.makedirs("model/weights", exist_ok=True)
    torch.save(generator.state_dict(), "model/weights/gan_generator_weights.pth")
    torch.save(discriminator.state_dict(), "model/weights/gan_discriminator_weights.pth")
    print("GAN Training complete. Weights saved.")
    
def cvae_loss_function(recon_x, x, mu, logvar, criterion):
    # Reshape for CrossEntropy
    recon_x_flat = recon_x.view(-1, recon_x.size(-1))
    x_flat = x.view(-1)
    
    recon_loss = criterion(recon_x_flat, x_flat)
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0) 
    
    # KL weight is kept small to allow the model to prioritize sequence reconstruction
    return recon_loss + (0.01 * kld)

def train_cvae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CVAE on device: {device}")
    
    tokenizer = DatesTokenizer()
    dataset = DatesDataset("data/data.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = ConditionalVAE(vocab_size=len(tokenizer.chars)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
    
    epochs = 15
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(x, y)
            
            loss = cvae_loss_function(recon_batch, y, mu, logvar, criterion)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    os.makedirs("model/weights", exist_ok=True)
    torch.save(model.state_dict(), "model/weights/cvae_weights.pth")
    print("CVAE Training complete. Weights saved.")

if __name__ == "__main__":
    train_cvae()