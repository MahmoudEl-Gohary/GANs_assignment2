import argparse
import torch
import os
from model.dataset import DatesTokenizer
from model.models import ConditionalVAE

def predict(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DatesTokenizer()
    
    # Initialize the model and load the trained weights
    model = ConditionalVAE(vocab_size=len(tokenizer.chars)).to(device)
    weights_path = "model/weights/cvae_weights.pth"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts or len(parts) < 4:
                continue
            
            # Extract the 4 conditions
            conditions = parts[:4]
            
            # Encode and move to device
            x = tokenizer.encode_input(*conditions).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Inference phase (x is None)
                out = model(x)
                
                # Get the most likely characters
                pred_indices = out.argmax(dim=-1)[0]
                date_str = tokenizer.decode_output(pred_indices)
            
            # Write to the output file matching the data.txt format
            conditions_str = " ".join(conditions)
            fout.write(f"{conditions_str} {date_str}\n")
            
    print(f"Predictions successfully written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dates based on calendar conditions.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input text file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output text file")
    
    args = parser.parse_args()
    predict(args.input, args.output)