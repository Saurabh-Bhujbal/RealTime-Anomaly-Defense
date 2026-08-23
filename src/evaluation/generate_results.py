import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT /
    "experiments" /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results = {
    "baseline": {
        "clean_accuracy": 98.94
    },

    "attacks": {
        "FGSM": {
            "0.05": 95.75,
            "0.10": 86.99,
            "0.20": 40.38
        },

        "PGD": {
            "0.05": 95.02,
            "0.10": 84.08,
            "0.20": 62.41
        }
    },

    "detector_purification": {
        "FGSM": {
            "0.05": 96.17,
            "0.10": 89.64,
            "0.20": 54.02
        },

        "PGD": {
            "0.05": 95.65,
            "0.10": 88.06,
            "0.20": 75.79
        }
    },

    "full_defense": {
        "FGSM": {
            "0.05": 96.18,
            "0.10": 90.04,
            "0.20": 59.45
        },

        "PGD": {
            "0.05": 95.76,
            "0.10": 89.41,
            "0.20": 80.16
        }
    },

    "eot_pgd": {
        "epsilon": 0.20,
        "attack_accuracy": 61.94,
        "defended_accuracy": 79.46,
        "detection_rate": 67.63
    }
}


output_file = (
    RESULTS_DIR /
    "project_results.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "✓ Final project results saved to:"
)

print(output_file)