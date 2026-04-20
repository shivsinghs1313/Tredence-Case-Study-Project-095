import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SelfPruningMLP
from engine import train_one_epoch, evaluate
from utils import set_seed, setup_logging, save_checkpoint

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_experiment_results(results_df: pd.DataFrame, graphs_dir: str):
    """Generates professional visualizations comparing the experiments."""
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy vs Lambda
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=results_df, x="lambda", y="test_accuracy", marker="o", linewidth=2.5, color="blue")
    plt.title("Test Accuracy vs Sparsity Penalty (Lambda)", fontsize=14, fontweight="bold")
    plt.xscale("log") # Useful if lambdas are e.g. 0.0001, 0.001
    plt.xlabel("Lambda Penalty (Log Scale)")
    plt.ylabel("Test Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "accuracy_vs_lambda.png"), dpi=300)
    plt.close()
    
    # 2. Sparsity vs Lambda
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=results_df, x="lambda", y="sparsity_percentage", marker="o", linewidth=2.5, color="red")
    plt.title("Sparsity Level vs Sparsity Penalty (Lambda)", fontsize=14, fontweight="bold")
    plt.xscale("log")
    plt.xlabel("Lambda Penalty (Log Scale)")
    plt.ylabel("Sparsity (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "sparsity_vs_lambda.png"), dpi=300)
    plt.close()

    # 3. Accuracy vs Sparsity tradeoff curve
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=results_df, x="sparsity_percentage", y="test_accuracy", hue="lambda", s=150, palette="viridis")
    plt.title("Accuracy vs. Sparsity Trade-off", fontsize=14, fontweight="bold")
    plt.xlabel("Sparsity Level (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(title="Lambda Penalty")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "accuracy_vs_sparsity.png"), dpi=300)
    plt.close()

    # 4. Parameters Remaining Bar Chart
    plt.figure(figsize=(10, 6))
    # Standardize lambda for x axis displaying
    results_df['lambda_str'] = results_df['lambda'].apply(lambda x: f"{x:.4f}")
    sns.barplot(data=results_df, x="lambda_str", y="active_params", palette="Blues_d")
    plt.title("Active Parameters Remaining by Lambda", fontsize=14, fontweight="bold")
    plt.xlabel("Lambda Penalty")
    plt.ylabel("Total Active Parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "active_parameters.png"), dpi=300)
    plt.close()

def run_experiment(lambda_val: float, config: dict, device: torch.device, logger):
    logger.info(f"--- Starting Experiment with lambda={lambda_val} ---")
    
    # Dataloaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    # Model definition
    model = SelfPruningMLP(
        in_features=config['model']['in_features'],
        hidden_layers=config['model']['hidden_layers'],
        out_features=config['model']['out_features'],
        dropout=config['model']['dropout_rate']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=config['training']['mixed_precision'])

    epochs = config['training']['epochs']
    init_temp = config['training']['initial_temp']
    final_temp = config['training']['final_temp']
    
    best_acc = 0.0
    final_metrics = {}

    for epoch in range(1, epochs + 1):
        # Temperature Annealing - Exponential Decay
        current_temp = init_temp * ((final_temp / init_temp) ** (epoch / epochs))

        train_loss, train_ce, train_sp, train_acc = train_one_epoch(
            model, trainloader, optimizer, criterion, device, scaler, lambda_val, current_temp
        )
        
        val_loss, val_acc, val_sparsity_metrics = evaluate(
            model, testloader, criterion, device, hard_threshold=None
        )
        
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch [{epoch}/{epochs}] T={current_temp:.3f} | " +
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | " +
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | " +
                        f"Sparsity: {val_sparsity_metrics['overall_sparsity']:.2f}%")

        # Save Best Model by Accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config['paths']['checkpoints_dir'], f"model_lambda_{lambda_val}.pth")
            save_checkpoint(model, optimizer, epoch, save_path, is_best=True)
            final_metrics = val_sparsity_metrics
            final_metrics['test_accuracy'] = best_acc

    logger.info(f"Experiment Complete. lambda={lambda_val} | Best Acc: {best_acc:.2f}% | Final Sparsity: {final_metrics['overall_sparsity']:.2f}%")
    
    return {
        "lambda": lambda_val,
        "test_accuracy": best_acc,
        "sparsity_percentage": final_metrics['overall_sparsity'],
        "total_params": final_metrics['total_params'],
        "active_params": final_metrics['active_params'],
        "params_removed": final_metrics['params_removed']
    }

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['training']['seed'])
    
    # Create required directories
    os.makedirs(config['paths']['outputs_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['paths']['graphs_dir'], exist_ok=True)
    
    logger = setup_logging(os.path.join(config['paths']['outputs_dir'], "training.log"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using compute device: {device}")

    # Sweeping through defined Lambda values
    results = []
    lambdas = config['training']['lambdas']
    
    for l_val in lambdas:
        res = run_experiment(l_val, config, device, logger)
        results.append(res)
        
    # Collate and Save Metrics
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(config['paths']['outputs_dir'], "metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"All experiments finished. Metrics compiled at {metrics_path}")
    print("\n--- Summary of Results ---")
    print(results_df[['lambda', 'test_accuracy', 'sparsity_percentage', 'active_params']])
    
    # Generate Plots
    logger.info("Generating evaluation charts...")
    plot_experiment_results(results_df, config['paths']['graphs_dir'])
    logger.info("Dashboard visualizations generated successfully.")

if __name__ == "__main__":
    main()
