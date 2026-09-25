import argparse
import sys
import os
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.gmdcn import GMDCN
from src.optimization.ssa import SalpSwarmOptimizer
from src.optimization.cuckoo_search import CuckooSearchOptimizer
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_val_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

def train_briefly(model, train_loader, lr, strength, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            ce_loss.backward()

            # clip mode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=strength)
            optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="Compare SSA and Cuckoo Search on GMDCN")
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fast-dev-run", action="store_true", help="Use a tiny subset of data to speed up")
    args = parser.parse_args()

    set_seed(42)
    device = config.DEVICE

    print("=" * 60)
    print(f"COMPARING SSA AND CUCKOO SEARCH (pop={args.pop_size}, iter={args.max_iter}, epochs={args.epochs})")
    print("=" * 60)

    train_images, train_labels, test_images, test_labels = load_mnist(config.DATA_DIR)
    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    if args.fast_dev_run:
        train_indices = random.sample(range(len(train_dataset)), 2000)
        test_indices = random.sample(range(len(test_dataset)), 500)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    bounds = [
        (1e-4, 1e-2),   # learning_rate
        (0.1, 0.5),     # dropout
        (0.1, 1.0),     # manipulation_strength
    ]

    eval_count = 0
    def fitness_fn(params):
        nonlocal eval_count
        eval_count += 1
        lr, dropout, strength = params
        model = GMDCN(in_channels=1, num_classes=config.NUM_CLASSES, dropout=dropout).to(device)
        train_briefly(model, train_loader, lr, strength, args.epochs, device)
        acc = evaluate_val_accuracy(model, val_loader, device)
        print(f"  Eval {eval_count} -> lr: {lr:.5f}, drop: {dropout:.3f}, str: {strength:.3f} => Acc: {acc:.2f}%")
        return acc

    print("\n--- Running SSA ---")
    eval_count = 0
    ssa = SalpSwarmOptimizer(fitness_fn, bounds, population_size=args.pop_size, max_iter=args.max_iter)
    ssa_best, ssa_score, ssa_hist = ssa.run()

    print("\n--- Running Cuckoo Search ---")
    eval_count = 0
    cuckoo = CuckooSearchOptimizer(fitness_fn, bounds, population_size=args.pop_size, max_iter=args.max_iter)
    cuckoo_best, cuckoo_score, cuckoo_hist = cuckoo.run()

    results = {
        "ssa": {
            "best_params": {
                "learning_rate": float(ssa_best[0]),
                "dropout": float(ssa_best[1]),
                "manipulation_strength": float(ssa_best[2])
            },
            "best_score": float(ssa_score),
            "convergence": [float(h) for h in ssa_hist]
        },
        "cuckoo": {
            "best_params": {
                "learning_rate": float(cuckoo_best[0]),
                "dropout": float(cuckoo_best[1]),
                "manipulation_strength": float(cuckoo_best[2])
            },
            "best_score": float(cuckoo_score),
            "convergence": [float(h) for h in cuckoo_hist]
        }
    }

    result_path = PROJECT_ROOT / "experiments" / "results" / "optimizer_comparison.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
        
    figure_dir = PROJECT_ROOT / "experiments" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(ssa_hist, label="Salp Swarm Algorithm (SSA)", color="royalblue", marker="o", linestyle="-", linewidth=2)
    plt.plot(cuckoo_hist, label="Cuckoo Search", color="darkorange", marker="s", linestyle="-", linewidth=2)
    plt.title("Optimizer Convergence Comparison: SSA vs. Cuckoo Search", fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=11)
    
    fig_path = figure_dir / "optimizer_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"SSA Best Score: {ssa_score:.2f}%")
    print(f"Cuckoo Best Score: {cuckoo_score:.2f}%")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {fig_path}")

if __name__ == "__main__":
    main()
