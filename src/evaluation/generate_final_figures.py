from pathlib import Path
import json

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD RESULTS
# ==========================================================

with open(
    RESULTS_DIR / "ablation_results.json",
    "r",
    encoding="utf-8"
) as f:
    ablation = json.load(f)


with open(
    RESULTS_DIR / "eot_results.json",
    "r",
    encoding="utf-8"
) as f:
    eot = json.load(f)


# ==========================================================
# FIGURE 1 — ABLATION FGSM
# ==========================================================

methods = [
    "Baseline",
    "Purification",
    "Randomized",
    "Gradient Diversity",
    "Combined"
]

method_keys = [
    "baseline",
    "purification",
    "randomized",
    "gradient",
    "combined"
]


for epsilon in ["0.10", "0.20"]:

    values = [
        ablation["FGSM"][epsilon][key]
        for key in method_keys
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        methods,
        values
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Defense Method")

    plt.title(
        f"FGSM Defense Ablation "
        f"(ε = {epsilon})"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(0, 100)

    plt.tight_layout()

    output = (
        FIGURES_DIR /
        f"fgsm_ablation_{epsilon}.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# FIGURE 2 — ABLATION PGD
# ==========================================================

for epsilon in ["0.10", "0.20"]:

    values = [
        ablation["PGD"][epsilon][key]
        for key in method_keys
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        methods,
        values
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Defense Method")

    plt.title(
        f"PGD Defense Ablation "
        f"(ε = {epsilon})"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(0, 100)

    plt.tight_layout()

    output = (
        FIGURES_DIR /
        f"pgd_ablation_{epsilon}.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# FIGURE 3 — BASELINE VS COMBINED
# ==========================================================

attacks = [
    "FGSM",
    "PGD"
]

epsilons = [
    "0.10",
    "0.20"
]


for epsilon in epsilons:

    baseline = [
        ablation[attack][epsilon]["baseline"]
        for attack in attacks
    ]

    combined = [
        ablation[attack][epsilon]["combined"]
        for attack in attacks
    ]

    x = range(len(attacks))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        label="Baseline"
    )

    plt.bar(
        [i + width / 2 for i in x],
        combined,
        width,
        label="Combined Defense"
    )

    plt.xticks(
        list(x),
        attacks
    )

    plt.ylabel("Accuracy (%)")

    plt.xlabel("Attack")

    plt.title(
        f"Baseline vs Combined Defense "
        f"(ε = {epsilon})"
    )

    plt.ylim(0, 100)

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES_DIR /
        f"baseline_vs_combined_{epsilon}.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# FIGURE 4 — ACCURACY RECOVERY
# ==========================================================

for attack in ["FGSM", "PGD"]:

    eps = ["0.10", "0.20"]

    baseline = [
        ablation[attack][e]["baseline"]
        for e in eps
    ]

    combined = [
        ablation[attack][e]["combined"]
        for e in eps
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        eps,
        baseline,
        marker="o",
        linewidth=2,
        label="Baseline"
    )

    plt.plot(
        eps,
        combined,
        marker="o",
        linewidth=2,
        label="Combined Defense"
    )

    plt.xlabel("Attack Strength (ε)")
    plt.ylabel("Accuracy (%)")

    plt.title(
        f"{attack} Robustness vs Attack Strength"
    )

    plt.ylim(0, 100)

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        FIGURES_DIR /
        f"{attack.lower()}_robustness_curve.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# FIGURE 5 — EOT RESULT
# ==========================================================

eot_attack = eot["attack_accuracy"]
eot_defended = eot["defended_accuracy"]

plt.figure(figsize=(8, 6))

plt.bar(
    ["EOT-PGD Attack", "Full Defense"],
    [
        eot_attack,
        eot_defended
    ]
)

plt.ylabel("Accuracy (%)")

plt.title(
    f"Adaptive EOT-PGD Defense "
    f"(ε = {eot['epsilon']})"
)

plt.ylim(0, 100)

plt.tight_layout()

output = (
    FIGURES_DIR /
    "eot_pgd_defense.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# FIGURE 6 — DETECTION RATE
# ==========================================================

detection_rates = [
    13.66,
    33.03,
    82.47,
    15.33,
    36.30,
    67.89
]

labels = [
    "FGSM\n0.05",
    "FGSM\n0.10",
    "FGSM\n0.20",
    "PGD\n0.05",
    "PGD\n0.10",
    "PGD\n0.20"
]

plt.figure(figsize=(10, 6))

plt.bar(
    labels,
    detection_rates
)

plt.ylabel("Detection Rate (%)")
plt.xlabel("Attack")

plt.title(
    "Anomaly Detection Rate"
)

plt.ylim(0, 100)

plt.tight_layout()

output = (
    FIGURES_DIR /
    "detection_rate.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPLETE
# ==========================================================

print("=" * 70)
print("FINAL FIGURE GENERATION COMPLETE")
print("=" * 70)

for file in sorted(FIGURES_DIR.glob("*.png")):

    print(
        f"✓ {file.name}"
    )

print()
print(
    f"Figures saved in:\n{FIGURES_DIR}"
)