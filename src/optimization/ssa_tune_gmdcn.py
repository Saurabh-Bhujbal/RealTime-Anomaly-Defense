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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.models.gmdcn import GMDCN
from src.optimization.ssa import SalpSwarmOptimizer
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
    parser = argparse.ArgumentParser(description="Tune GMDCN using SSA")
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fast-dev-run", action="store_true", help="Use a tiny subset of data to speed up")
    args = parser.parse_args()

    set_seed(42)
    device = config.DEVICE

    print("=" * 60)
    print(f"SSA TUNING GMDCN (pop={args.pop_size}, iter={args.max_iter}, epochs={args.epochs})")
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
        print(f"Eval {eval_count} -> lr: {lr:.5f}, drop: {dropout:.3f}, str: {strength:.3f} => Acc: {acc:.2f}%")
        return acc

    optimizer = SalpSwarmOptimizer(fitness_fn, bounds, population_size=args.pop_size, max_iter=args.max_iter)
    best_params, best_score, history = optimizer.run()

    results = {
        "best_params": {
            "learning_rate": float(best_params[0]),
            "dropout": float(best_params[1]),
            "manipulation_strength": float(best_params[2])
        },
        "best_score": float(best_score),
        "convergence": [float(h) for h in history]
    }

    result_path = PROJECT_ROOT / "experiments" / "results" / "ssa_tuning_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best Score: {best_score:.2f}%")
    print(f"Best Params: {results['best_params']}")
    print(f"Results saved to: {result_path}")

if __name__ == "__main__":
    main()
