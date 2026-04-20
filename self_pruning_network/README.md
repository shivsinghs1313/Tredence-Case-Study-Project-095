# The Self-Pruning Neural Network 🧠✂️

A complete PyTorch pipeline demonstrating a custom neural network architecture that learns to dynamically disable its own weak weight connections during training. 

By applying a sparsity-inducing L1 penalty on learnable structural gates, the network achieves a highly sparse representation without sacrificing significant accuracy.

## What Problem This Solves
Modern Large Language Models (LLMs) and deep vision models are heavily overparameterized. While necessary for the optimization landscape during training, this overparameterization wastes memory, power, and compute during inference.

## Why Dynamic Pruning is Better Than Post-Training Pruning
Traditional pruning involves training a dense network, stopping, sorting weights by magnitude, deleting the smallest ones, and optionally fine-tuning. 
Dynamic pruning (Self-Pruning via Gating) allows the network to incorporate the compression goals directly into its optimization trajectory. The network "knows" it needs to be small and can intelligently route information through a few critical paths early in training, avoiding the shock of sudden magnitude-based surgery.

## How The Architecture Works
We designed a custom PyTorch layer, `PrunableLinear`. For every standard `weight` connecting neurons, there is a parallel `gate_score` parameter.
The forward pass is calculated as:
```python
gates = torch.sigmoid(gate_scores / temperature)
effective_weight = weight * gates
output = F.linear(input, effective_weight, bias)
```
As temperature anneals toward zero during training, the `gates` are forced to be binary (0.0 or 1.0). The loss function penalizes nonzero gates:
`Total Loss = CrossEntropyLoss + λ * sum(gates)`

## How to Run Training
This project requires Python 3.11+ and GPU (CUDA) support is highly recommended but not strictly required.
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the main experiment suite
# This automatically runs training for 5 lambda values [0, 0.0001, 0.001, 0.005, 0.01]
python train.py --config config.yaml
```

Outputs will be saved in:
- `outputs/metrics.csv`: Data logging test accuracy, remaining params, and sparsity %.
- `outputs/graphs/`: Professional visualizations of tradeoffs.
- `outputs/checkpoints/`: Model weights.

### Exporing to Hard Inference Mode
To zero out sparse weights for real:
```bash
python export.py --checkpoint outputs/checkpoints/model_lambda_0.001.pth
```
This generates an ONNX compatible `.onnx` graph and a `_hardened.pth` model.

### Latency Benchmarking
To test inference speed vs dense baseline:
```bash
python inference.py --checkpoint outputs/checkpoints/model_lambda_0.001.pth
```

## Sample Results Expectation (CIFAR-10)
| Sparsity Penalty (λ) | Test Accuracy | Sparsity % | Active Parameters |
|----------------------|----------------|------------|--------------------|
| 0.0 (Dense)          | 58.2%          | 0.0%       | 8,924,170         |
| 0.0001               | 57.8%          | 32.5%      | ~6,000,000        |
| 0.001                | 54.1%          | 85.0%      | ~1,300,000        |
| 0.01                 | 40.0%          | 98.2%      | <150,000          |

*(Metrics vary based on epochs and hyperparameters. Run `train.py` for exact figures).*

## Real-World Applications

1. **Mobile AI Apps**: Compressing vision or language models to fit under the strict 50MB-100MB RAM budgets of iOS and Android mobile apps. Self-pruning allows us to ship smaller `.tflite` or CoreML artifacts.
2. **Edge Devices / IoT**: Deploying models on microcontrollers (Arduino, Raspberry Pi) requiring low latency and low power usage limit parameter count drastically.
3. **Real-time Systems**: High FPS demands in autonomous driving or live video streaming detection where memory bandwidth is the primary bottleneck.
4. **Cloud Cost Reduction**: Smaller parameter loads mean higher batch throughput on expensive H100s, drastically cutting API serving costs per token/image.
5. **Sustainable AI**: Sparse models require fewer FLOPs if deployed intelligently via block-sparse ops, generating lower heat and energy strain.

## Future Upgrades
- Implement unstructured `torch.sparse` conversion to leverage actual GPU memory savings instead of just mask simulation.
- Introduce block-wise gating to match modern GPU Tensor Core alignments (e.g., N:M sparsity).
- Extend the `PrunableConvolution` layer for CNNs.
