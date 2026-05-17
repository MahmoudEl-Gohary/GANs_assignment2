import torch
import random
import numpy as np

def init_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print("GPU optimizations applied.")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    init_environment()