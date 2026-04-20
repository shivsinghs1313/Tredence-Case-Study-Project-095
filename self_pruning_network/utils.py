import os
import random
import numpy as np
import torch
import logging
from typing import Dict, Any
import yaml

def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file: str = "training.log") -> logging.Logger:
    """Configures the logging system."""
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("PruningLogger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Add file handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def save_checkpoint(model, optimizer, epoch: int, path: str, is_best: bool = False):
    """Saves model checkpoints."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)
    if is_best:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(state, best_path)

def calculate_sparsity(model, hard_threshold: float = 0.01) -> Dict[str, Any]:
    """Calculates overall and layer-wise sparsity based on gate values."""
    total_params = 0
    zero_params = 0
    layer_sparsity = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'gate_scores'):
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores)
                
                # Active if gate is strictly greater than the threshold
                active = (gates > hard_threshold).sum().item()
                total = gates.numel()
                zero = total - active
                
                total_params += total
                zero_params += zero
                
                layer_sparsity[name] = zero / total if total > 0 else 0
                
    overall_sparsity = (zero_params / total_params) if total_params > 0 else 0.0
    return {
        "overall_sparsity": overall_sparsity * 100.0, # percentage
        "total_params": total_params,
        "active_params": total_params - zero_params,
        "layer_sparsity": layer_sparsity,
        "params_removed": zero_params
    }
