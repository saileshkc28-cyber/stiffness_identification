"""Run one stage of a case.

Usage: python run_case.py <case> <check|phase1|infer>
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = {
    "check": ("bayesian_inference", "check_forward.py"),
    "phase1": ("damaged_system", "MainKratos_phase1.py"),
    "infer": ("bayesian_inference", "MainBayesian.py"),
}


def use_case_sigma(case_dir):
    with open(os.path.join(case_dir, "noise_sigma.json")) as f:
        sigma = json.load(f)["sigma"]
    path = os.path.join(case_dir, "bayesian_inference", "BayesianParameters.json")
    with open(path) as f:
        settings = json.load(f)
    settings["likelihood"]["noise_model"]["sigma"] = sigma
    with open(path, "w") as f:
        json.dump(settings, f, indent=4)


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in STAGES:
        sys.exit(__doc__)

    case, stage = sys.argv[1], sys.argv[2]
    case_dir = os.path.join(ROOT, "cases", case)
    if not os.path.isdir(case_dir):
        sys.exit(f"no case folder: {case_dir}")

    sigma_file = os.path.join(case_dir, "noise_sigma.json")
    data_file = os.path.join(case_dir, "damaged_system", "measured_data.csv")
    if stage in ("phase1", "infer") and not os.path.exists(sigma_file):
        sys.exit(f"run 'python run_case.py {case} check' first: no noise_sigma.json")
    if stage == "infer":
        if not os.path.exists(data_file):
            sys.exit(f"run 'python run_case.py {case} phase1' first: no measured_data.csv")
        use_case_sigma(case_dir)

    folder, script = STAGES[stage]
    result = subprocess.run([sys.executable, os.path.join(ROOT, "core", script)],
                            cwd=os.path.join(case_dir, folder))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
