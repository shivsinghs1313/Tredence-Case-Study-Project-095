import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrunableLinear(nn.Module):
    """
    Custom Linear Layer with Learnable Structural Gates.
    Each weight connection has a corresponding learnable gate_score.
    """
    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using Kaiming normal and gates to positive values."""
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gates with positive values so they start near 1.0 (active)
        # This allows the network to begin fully connected and prune over time
        nn.init.constant_(self.gate_scores, 1.5)

    def forward(self, input: torch.Tensor, temperature: float = 1.0, hard_threshold: float = None) -> torch.Tensor:
        """
        Forward pass applying temperature-controlled sigmoid gating.
        If hard_threshold is set (e.g., during export/inference), exactly zeroes out weak weights.
        """
        if hard_threshold is not None:
            # During hard inference, discretize to 0.0 or 1.0 based on threshold
            gates = (torch.sigmoid(self.gate_scores) >= hard_threshold).float()
        else:
            # During training, use temperature annealing
            gates = torch.sigmoid(self.gate_scores / temperature)
            
        effective_weight = self.weight * gates
        return F.linear(input, effective_weight, self.bias)


class SelfPruningMLP(nn.Module):
    """
    Modern MLP classifier for CIFAR-10 that dynamically prunes itself during training.
    Architecture: Flatten -> PrunableLinear blocks -> Classifier -> output
    Default: 3072 -> 2048 -> 1024 -> 512 -> 10
    """
    def __init__(self, in_features: int = 3072, hidden_layers: list = [2048, 1024, 512], out_features: int = 10, dropout: float = 0.1):
        super(SelfPruningMLP, self).__init__()
        
        layers = []
        current_in = in_features
        
        for h in hidden_layers:
            layers.append(PrunableLinear(current_in, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU()) # Modern activation (GELU > ReLU)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_in = h
            
        # Final fully connected classifier
        layers.append(PrunableLinear(current_in, out_features))
        
        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard_threshold: float = None) -> torch.Tensor:
        # Flatten CIFAR-10 images (B, C, H, W) -> (B, C*H*W)
        x = x.view(x.shape[0], -1)
        
        for layer in self.network:
            if isinstance(layer, PrunableLinear):
                x = layer(x, temperature=temperature, hard_threshold=hard_threshold)
            else:
                x = layer(x)
        return x
        
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Computes the sum of all sigmoid gate scores for the sparsity penalty.
        Minimizing this pushes the network to turn off connections (gates -> 0).
        """
        l1_penalty = 0.0
        for layer in self.network:
            if isinstance(layer, PrunableLinear):
                # Standard L1 on gate outputs
                l1_penalty += torch.sum(torch.sigmoid(layer.gate_scores))
        return l1_penalty
