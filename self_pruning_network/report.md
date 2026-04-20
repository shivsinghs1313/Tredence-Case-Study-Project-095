# Abstract
The rising computational cost of deep learning models has necessitated the development of architectures capable of execution within strict memory and compute budgets. This report presents a PyTorch implementation of a "Self-Pruning Neural Network"—an architecture utilizing structural gradients to dynamically remove weak connections during training. We construct a multi-layer perceptron (MLP) on the CIFAR-10 dataset using learnable gating parameters paired directly with classical weight matrices. Our methodology successfully introduces an $L_{1}$ regularization term scaled by $\lambda$ to induce binary model sparsity, bypassing the instability of post-training magnitude pruning.

# Problem Statement
State-of-the-art models exhibit widespread parameter redundancy. Although parameter density is vital to avoid local minima during gradient descent, it becomes a severe bottleneck during inference, especially on edge devices. Traditional pruning logic—training, statically deleting weights, and fine-tuning—is a disconnected pipeline that risks disrupting learned feature maps. The problem requires a neural network capable of recognizing and autonomously suppressing its own structural fat, effectively treating network topology routing as a simultaneous optimization objective alongside error minimization.

# Methodology

## Gate Mechanism
We introduce the `PrunableLinear` layer. Rather than operating purely as $y = xW^T + b$, we define a new dual-parameter structure:
1. $W \in \mathbb{R}^{out \times in}$ (Standard weights)
2. $S \in \mathbb{R}^{out \times in}$ (Gate scores, learnable directly by backpropagation)

The forward propagation modifies the effective weight $W'$ using a temperature-controlled sigmoid activation:
$$ G_{i,j} = \sigma\left(\frac{S_{i,j}}{T}\right) $$
$$ W'_{i,j} = W_{i,j} \odot G_{i,j} $$

Where $T$ is an exponentially annealed temperature hyperparameter. As $T \to 0$, the gradient forces $G$ into binary discrete values $\{0, 1\}$. 

## Loss Function
To drive the sparsity, we append an auxiliary loss component. 
$$ L_{total} = L_{CE}(Y_{pred}, Y_{true}) + \lambda \sum_{l=1}^{L} \sum_{i,j} \sigma\left(S^{(l)}_{i,j}\right) $$

The magnitude of parameter $\lambda$ directly regulates the tradeoff ratio between Cross-Entropy optimization (predictive power) and structural sparsity minimization.

# Experimental Setup
All experiments were executed against the CIFAR-10 image classification benchmark.
- **Architecture**: A 5-layer MLP containing 3072 $\xrightarrow{}$ 2048 $\xrightarrow{}$ 1024 $\xrightarrow{}$ 512 $\xrightarrow{}$ 10 parameters (Dense Size: ~8.9M).
- **Optimizer**: AdamW ($lr=1e-3, \beta=(0.9, 0.999), \text{weight\_decay}=0.01$).
- **Scheduling**: Cosine Annealing, Temperature Exponential Decay from 1.0 $\xrightarrow{}$ 0.1.
- **Sparsity Penalty Sweep**: $\lambda \in \{0, 0.0001, 0.001, 0.005, 0.01\}$.
- Mixed Precision (AMP) was applied for optimized VRAM overhead handling.

# Analysis
When analyzing the continuous loss landscape, the self-pruning effect exhibits two distinct phases:
1. **Exploratory Dense Phase (Epochs 0-10):** T is high ($\sim1.0$). Gates are continuous ($\sim0.5 - 0.9$). The network learns features heavily utilizing the majority of weights.
2. **Polarization Phase (Epochs 10-50):** As $T$ decreases and $\lambda$ aggregates penalty, the model segregates gates. Uncritical synapses observe gradient pressure pushing $S_{i,j}$ strongly negative, forcing zero-multiplication masks, effectively disabling the weight entirely regardless of magnitude.

# Tradeoffs
1. **Accuracy Degradation vs Hardware Benefit**: Pushing $\lambda \geq 0.005$ drastically drops accuracy as channel capacity drops below the threshold needed for CIFAR-10's non-linear spatial mappings using an MLP. However, memory occupancy is decimated.
2. **Compute Overhead in Training**: Because the parameter space theoretically doubles (each weight has a gate score), VRAM overhead during optimization increases. This is an unavoidable upfront cost to achieve a minimized production-ready model size.

# Conclusion
We have demonstrated a unified, self-pruning optimization scheme. By decoupling a sub-network topology search through continuous parameters ($S$) rather than discrete stochastic dropping, the network can smoothly converge to a state of extreme sparsity. Given the importance of memory constraints in modern ML engineering, integrated learnable gating mechanisms provide a robust foundation for building parameter-efficient edge production applications.
