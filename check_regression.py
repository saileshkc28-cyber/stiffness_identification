"""Compare a run's summary.json with a saved reference summary.

Usage: python check_regression.py <summary.json> <reference.json>
"""
import json
import sys

TOLERANCE = 1e-6

with open(sys.argv[1]) as f:
    result = json.load(f)
with open(sys.argv[2]) as f:
    reference = json.load(f)

passed = len(result["zones"]) == len(reference["zones"])
print(f"zones: {len(result['zones'])} vs {len(reference['zones'])}")

for run, ref in zip(result["zones"], reference["zones"]):
    for key in ("alpha_mean", "alpha_std"):
        difference = abs(run[key] - ref[key])
        passed = passed and difference <= TOLERANCE
        print(f"zone {ref['zone']} {key}: {run[key]:.9f} vs {ref[key]:.9f} "
              f"(difference {difference:.2e})")

passed = passed and result["n_forward_solves"] == reference["n_forward_solves"]
print(f"forward solves: {result['n_forward_solves']} vs {reference['n_forward_solves']}")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
