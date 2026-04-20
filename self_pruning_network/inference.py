import time
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SelfPruningMLP
from engine import evaluate
from utils import set_seed, calculate_sparsity

def measure_inference_speed(model, dataloader, device, hard_threshold=0.01):
    """
    Measures the inference latency per batch and checks accuracy
    using the hard threshold logic (export mode).
    """
    model.eval()
    times = []
    
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    print("Warming up GPU/CPU...")
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= 5: break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs, hard_threshold=hard_threshold)
            
    # Measure
    print("Measuring inference...")
    loss, val_acc, val_sparsity_metrics = evaluate(model, dataloader, criterion, device, hard_threshold=hard_threshold)
    
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            if device.type == 'cuda':
                start_event.record()
                _ = model(inputs, hard_threshold=hard_threshold)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event)) # Returns in ms
            else:
                s = time.time()
                _ = model(inputs, hard_threshold=hard_threshold)
                e = time.time()
                times.append((e - s) * 1000) # ms
                
    avg_latency = sum(times) / len(times)
    
    print(f"\n--- Inference Benchmark Results ---")
    print(f"Device: {device}")
    print(f"Hard Threshold: {hard_threshold}")
    print(f"Test Accuracy: {val_acc:.2f}%")
    print(f"Sparsity: {val_sparsity_metrics['overall_sparsity']:.2f}%")
    print(f"Total Params Remaining: {val_sparsity_metrics['active_params']} / {val_sparsity_metrics['total_params']}")
    print(f"Average Batch Latency: {avg_latency:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.01, help="Hard structural gate threshold")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['training']['seed'])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config['data']['batch_size'], shuffle=False)

    model = SelfPruningMLP(
        in_features=config['model']['in_features'],
        hidden_layers=config['model']['hidden_layers'],
        out_features=config['model']['out_features']
    ).to(device)

    # Load Weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # the checkpointer saves state_dict inside "state_dict" key if coming from our train.py
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    measure_inference_speed(model, testloader, device, hard_threshold=args.threshold)

if __name__ == "__main__":
    main()
