import torch
import torch.nn as nn
from tqdm import tqdm
from utils import calculate_sparsity

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, lambda_val: float, temperature: float):
    """Trains the model for one epoch including the sparsity penalty."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_sparsity_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed Precision Training
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=scaler.is_enabled()):
            outputs = model(inputs, temperature=temperature)
            ce_loss = criterion(outputs, targets)
            sparsity_penalty = model.get_sparsity_loss()
            loss = ce_loss + (lambda_val * sparsity_penalty)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Tracking metrics
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_sparsity_loss += sparsity_penalty.item() * batch_size
        
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%", "CE": f"{ce_loss.item():.4f}"})
        
    epoch_loss = total_loss / total
    epoch_ce = total_ce_loss / total
    epoch_acc = 100. * correct / total
    epoch_sparsity = total_sparsity_loss / total
    
    return epoch_loss, epoch_ce, epoch_sparsity, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, hard_threshold=None):
    """Evaluates the model on validation/test set, returning metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        # Fix temperature to 1.0 for evaluation, or use hard_threshold
        outputs = model(inputs, temperature=1.0, hard_threshold=hard_threshold)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    
    # Calculate sparsity metrics
    sparsity_metrics = calculate_sparsity(model, hard_threshold=hard_threshold if hard_threshold else 0.01)
    
    return epoch_loss, epoch_acc, sparsity_metrics
