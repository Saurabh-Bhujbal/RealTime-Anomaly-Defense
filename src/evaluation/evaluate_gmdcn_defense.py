import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gmdcn import GMDCN
from src.data.mnist_loader import load_mnist
from src.data.preprocessing import MNISTDataset
from src.data.cifar10_dataset import load_cifar10, CIFAR10Dataset
import config

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.cw import carlini_wagner_attack

try:
    from src.attacks.eot_pgd import eot_pgd_attack
    EOT_AVAILABLE = True
except ImportError:
    EOT_AVAILABLE = False

from src.detection.anomaly_detector import DynamicAnomalyDetector
from src.purification.purifier import SelfPurifier
from src.defenses.randomized_defense import RandomizedDefense
from src.defenses.gradient_diversity import GradientDiversityDefense
from src.defenses.combined_defense import CombinedDefense

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_defenses(model, calibration_loader):
    print("\nCalibrating anomaly detector...")
    detector = DynamicAnomalyDetector(model=model, threshold_percentile=95.0)
    threshold = detector.calibrate(clean_loader=calibration_loader, device=DEVICE)
    print(f"[OK] Anomaly threshold: {threshold:.6f}")

    purifier = SelfPurifier(model=model)
    randomized_defense = RandomizedDefense(model=model)
    gradient_defense = GradientDiversityDefense(model=model)
    
    combined_defense = CombinedDefense(
        model=model,
        detector=detector,
        purifier=purifier,
        randomized_defense=randomized_defense,
        gradient_defense=gradient_defense
    )
    
    print("[OK] Purifier ready\n[OK] Randomized defense ready\n[OK] Gradient diversity ready\n[OK] Combined defense ready")
    return detector, purifier, randomized_defense, gradient_defense, combined_defense, threshold

def evaluate_defenses(model, detector, purifier, randomized_defense, gradient_defense, combined_defense, images, labels):
    with torch.no_grad():
        logits = model(images)
        baseline_predictions = logits.argmax(dim=1)
    baseline_accuracy = (baseline_predictions == labels).float().mean().item() * 100

    detection = detector.detect(images)
    anomalous = detection["is_anomalous"]
    detection_rate = anomalous.float().mean().item() * 100

    purified_images = images.clone()
    if anomalous.any():
        purified = purifier.purify(images[anomalous])
        purified_images[anomalous] = purified

    with torch.no_grad():
        purification_logits = model(purified_images)
        purification_predictions = purification_logits.argmax(dim=1)
    purification_accuracy = (purification_predictions == labels).float().mean().item() * 100

    randomized_result = randomized_defense.predict(purified_images)
    randomized_accuracy = (randomized_result["predictions"] == labels).float().mean().item() * 100

    gradient_result = gradient_defense.predict(purified_images)
    gradient_accuracy = (gradient_result["final_predictions"] == labels).float().mean().item() * 100

    combined_result = combined_defense.predict(images)
    combined_accuracy = (combined_result["predictions"] == labels).float().mean().item() * 100

    return {
        "baseline": baseline_accuracy,
        "purification": purification_accuracy,
        "randomized": randomized_accuracy,
        "gradient_diversity": gradient_accuracy,
        "combined": combined_accuracy,
        "detection_rate": detection_rate
    }

def run_attack(model, loader, attack_name, attack_function, attack_kwargs, defenses, max_batches=20):
    detector, purifier, randomized_defense, gradient_defense, combined_defense, threshold = defenses
    totals = {k: 0.0 for k in ["baseline", "purification", "randomized", "gradient_diversity", "combined", "detection_rate"]}
    total_batches = 0
    total_samples = 0
    
    print(f"\n{'-'*80}\n{attack_name}\n{'-'*80}")
    
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= max_batches: break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        adversarial_images = attack_function(model, images, labels, **attack_kwargs)
        metrics = evaluate_defenses(model, detector, purifier, randomized_defense, gradient_defense, combined_defense, adversarial_images, labels)
        
        batch_size = labels.size(0)
        for key in totals: totals[key] += metrics[key] * batch_size
        total_samples += batch_size
        total_batches += 1
        print(f"{attack_name}: batch {batch_index + 1}/{max_batches}")

    final_metrics = {k: v / total_samples for k, v in totals.items()}
    print(f"\nBaseline           : {final_metrics['baseline']:.2f}%")
    print(f"Purification       : {final_metrics['purification']:.2f}%")
    print(f"Randomized         : {final_metrics['randomized']:.2f}%")
    print(f"Gradient Diversity : {final_metrics['gradient_diversity']:.2f}%")
    print(f"Combined           : {final_metrics['combined']:.2f}%")
    print(f"Detection Rate     : {final_metrics['detection_rate']:.2f}%")
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate GMDCN defenses")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist", "cifar10"], default="mnist")
    parser.add_argument("--manipulation-mode", type=str, choices=["clip", "mask", "penalty"], default="clip")
    args = parser.parse_args()

    print("=" * 80)
    print(f"GMDCN MODEL ({args.manipulation_mode}) + DEFENSE EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    model_path = os.path.join(PROJECT_ROOT, "models", f"gmdcn_cnn_{args.manipulation_mode}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GMDCN model not found:\n{model_path}")

    print(f"\nLoading existing {args.dataset} test data...")
    if args.dataset in ["mnist", "fashion_mnist"]:
        in_channels = 1
        data_dir = config.DATA_DIR if args.dataset == "mnist" else config.FASHION_MNIST_DIR
        train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
        dataset = MNISTDataset(test_images, test_labels)
        calibration_dataset = MNISTDataset(train_images, train_labels)
    elif args.dataset == "cifar10":
        in_channels = 3
        cifar_dir = os.path.join(PROJECT_ROOT, "data", "cifar10")
        train_images, train_labels, test_images, test_labels = load_cifar10(cifar_dir)
        dataset = CIFAR10Dataset(test_images, test_labels)
        calibration_dataset = CIFAR10Dataset(train_images, train_labels)

    calibration_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
    evaluation_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"[OK] Test samples: {len(dataset)}")

    model = GMDCN(in_channels=in_channels, num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("[OK] GMDCN CNN loaded")

    defenses = create_defenses(model, calibration_loader)
    threshold = defenses[-1]

    results = {
        "model": f"GMDCN CNN ({args.manipulation_mode})",
        "dataset": args.dataset,
        "device": DEVICE,
        "anomaly_threshold": threshold,
        "fgsm": {},
        "pgd": {},
        "cw": {},
        "eot_pgd": {}
    }

    for epsilon in [0.10, 0.20]:
        metrics = run_attack(model, evaluation_loader, f"FGSM epsilon={epsilon:.2f}", fgsm_attack, {"epsilon": epsilon}, defenses)
        results["fgsm"][str(epsilon)] = metrics

    for epsilon in [0.10, 0.20]:
        metrics = run_attack(model, evaluation_loader, f"PGD epsilon={epsilon:.2f}", pgd_attack, {"epsilon": epsilon, "alpha": 0.01, "steps": 10}, defenses)
        results["pgd"][str(epsilon)] = metrics

    results["cw"] = run_attack(model, evaluation_loader, "C&W", carlini_wagner_attack, {"targeted": False, "c": 1.0, "kappa": 0.0, "lr": 0.01, "steps": 100, "device": DEVICE}, defenses)

    if EOT_AVAILABLE:
        try:
            results["eot_pgd"] = run_attack(model, evaluation_loader, "EOT-PGD", eot_pgd_attack, {"epsilon": 0.20, "alpha": 0.01, "steps": 10}, defenses)
        except TypeError as error:
            print(f"\n⚠ EOT-PGD skipped: {error}")

    result_path = os.path.join(PROJECT_ROOT, "experiments", "results", f"gmdcn_defense_results_{args.manipulation_mode}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {result_path}")

if __name__ == "__main__":
    main()
