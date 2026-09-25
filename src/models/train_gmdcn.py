import argparse
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.data.cifar10_dataset import load_cifar10, CIFAR10Dataset
from src.models.gmdcn import GMDCN
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
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

def main():
    parser = argparse.ArgumentParser(description="Train GMDCN with gradient manipulation")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist", "cifar10"], default="mnist")
    parser.add_argument("--manipulation-mode", type=str, choices=["clip", "mask", "penalty"], default="clip")
    parser.add_argument("--manipulation-strength", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = config.DEVICE

    print("=" * 60)
    print(f"GMDCN TRAINING (Dataset: {args.dataset}, Mode: {args.manipulation_mode}, Strength: {args.manipulation_strength})")
    print("=" * 60)
    print(f"Device: {device}")

    if args.dataset in ["mnist", "fashion_mnist"]:
        in_channels = 1
        data_dir = config.DATA_DIR if args.dataset == "mnist" else config.FASHION_MNIST_DIR
        train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
        train_dataset = MNISTDataset(train_images, train_labels)
        test_dataset = MNISTDataset(test_images, test_labels)
    elif args.dataset == "cifar10":
        in_channels = 3
        cifar_dir = PROJECT_ROOT / "data" / "cifar10"
        train_images, train_labels, test_images, test_labels = load_cifar10(cifar_dir)
        train_dataset = CIFAR10Dataset(train_images, train_labels)
        test_dataset = CIFAR10Dataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = GMDCN(in_channels=in_channels, num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if args.manipulation_mode == "penalty":
                images.requires_grad_(True)
            
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            if args.manipulation_mode == "penalty":
                input_grad = torch.autograd.grad(ce_loss, images, create_graph=True)[0]
                grad_penalty = input_grad.norm(p=2, dim=(1,2,3)).mean()
                loss = ce_loss + args.manipulation_strength * grad_penalty
                loss.backward()
                optimizer.step()
            else:
                loss = ce_loss
                loss.backward()

                if args.manipulation_mode == "clip":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.manipulation_strength)
                elif args.manipulation_mode == "mask":
                    for p in model.parameters():
                        if p.grad is None: continue
                        flat = p.grad.abs().flatten()
                        k = int(args.manipulation_strength * flat.numel())
                        if k > 0:
                            threshold = flat.kthvalue(k).values
                            p.grad[p.grad.abs() < threshold] = 0.0
                
                optimizer.step()

            running_loss += loss.item()

        train_accuracy = evaluate(model, train_loader, device)
        test_accuracy = evaluate(model, test_loader, device)
        average_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{config.EPOCHS}] Loss: {average_loss:.4f} Train Acc: {train_accuracy:.2f}% Test Acc: {test_accuracy:.2f}%")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / f"gmdcn_cnn_{args.manipulation_mode}.pth"
    torch.save(model.state_dict(), model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
