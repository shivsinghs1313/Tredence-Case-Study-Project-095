import argparse
import os
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from model import SelfPruningMLP

def plot_weight_distributions(dense_model, pruned_model, out_dir):
    """Generates histograms of parameter values to demonstrate pruning effect."""
    print("Generating weight distribution visualizations...")
    dense_weights = []
    pruned_weights = []
    
    for name, module in dense_model.named_modules():
        if hasattr(module, 'weight') and hasattr(module, 'gate_scores'): # is PrunableLinear
            dense_weights.extend(module.weight.data.view(-1).cpu().numpy())
            
    for name, module in pruned_model.named_modules():
         if hasattr(module, 'weight') and hasattr(module, 'gate_scores'): # is PrunableLinear
            gate_mask = (torch.sigmoid(module.gate_scores) >= 0.01).float()
            effective_weight = module.weight.data * gate_mask
            pruned_weights.extend(effective_weight.view(-1).cpu().numpy())

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(dense_weights, bins=100, color="blue", log_scale=(False, True))
    plt.title("Before Overwrite: Raw Weights")
    plt.xlabel("Weight Value")
    
    plt.subplot(1, 2, 2)
    sns.histplot(pruned_weights, bins=100, color="red", log_scale=(False, True))
    plt.title("After Pruning Hard Threshold")
    plt.xlabel("Weight Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weight_distribution.png"), dpi=300)
    plt.close()


def export_hard_model(checkpoint_path: str, config: dict, threshold: float = 0.01):
    """
    Bakes the gate_scores into the raw weights directly and removes gates.
    The resulting structure simulates what would be shipped to an inference server.
    """
    device = torch.device('cpu')
    model = SelfPruningMLP(
        in_features=config['model']['in_features'],
        hidden_layers=config['model']['hidden_layers'],
        out_features=config['model']['out_features']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create a shadow copy of the model to overwrite its weights with hardened sparse tensors
    import copy
    baked_model = copy.deepcopy(model)

    for name, module in baked_model.named_modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            mask = (gates >= threshold).float()
            # Overwrite raw weights
            module.weight.data = module.weight.data * mask

    os.makedirs(config['paths']['graphs_dir'], exist_ok=True)
    plot_weight_distributions(model, baked_model, config['paths']['graphs_dir'])

    export_path = checkpoint_path.replace(".pth", "_hardened.pth")
    torch.save(baked_model.state_dict(), export_path)
    print(f"Exported hard-thresholded sparse model to {export_path}")
    
    # Optional ONNX Export
    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        onnx_path = export_path.replace(".pth", ".onnx")
        torch.onnx.export(baked_model, dummy_input, onnx_path, 
                          export_params=True, 
                          opset_version=14, 
                          do_constant_folding=True,
                          input_names=['input'], 
                          output_names=['output'])
        print(f"Exported to ONNX at {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed/skipped: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export model to hard threshold structure")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    export_hard_model(args.checkpoint, config, args.threshold)

if __name__ == "__main__":
    main()
