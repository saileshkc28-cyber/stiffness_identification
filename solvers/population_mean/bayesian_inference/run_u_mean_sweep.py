r"""Mean-only (u_mean) sensor-count sweep. Edits nothing in the repository.

Phase 2 is fed the population-mean response of the 10-sensor Phase 1 dataset,
restricted to the first k sensors (tip first, then inward), for k in 1, 4, 10.
Every input a run needs is written into its own output folder:

    <archive>/N04/sensor_data.json          first 4 sensors of sensor_data_10s.json
    <archive>/N04/measured_data.csv         first 4 rows of measured_data_collapsed_clean.csv
    <archive>/N04/BayesianParameters.json   BayesianParameters.json pointed at the two
                                            files above, sigma 3.788e-08, batch off,
                                            every output process writing into N04/
    <archive>/N04/run_info.json             k, stations, sigma, clean mean u_true,
                                            N_eff and the two truth values
    <archive>/N04/phase2.log                MainBayesian.py console output

The repo's sensor_data.json and BayesianParameters.json are only read.

Noise model, option A: one instrument sigma = 3.788e-08 for every sensor, as in the
sensor, weighted and hierarchical sweeps.

Truth values, both from the valid Phase 1 specimens:
    alpha_pop_mean       mean(alpha_i)
    alpha_harmonic_mean  1 / mean(1 / alpha_i)   -- what the mean response encodes,
                                                    since u_i = u_i_ref / alpha

Prerequisite (no Kratos), from damaged_system/:
    python clean_phase1.py phase1_distribution_runs_10s
    python collapse_phase1.py phase1_distribution_runs_10s --sensors ../sensor_placement/sensor_data_10s.json

Run from bayesian_inference/:
    python run_u_mean_sweep.py                 # N01, N04, N10
    python run_u_mean_sweep.py N04             # re-run single cases
    python run_u_mean_sweep.py --prepare-only  # write inputs, skip MainBayesian.py
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\sensors_u_mean")
DATASET = HERE.parent / "damaged_system" / "phase1_distribution_runs_10s"
SENSORS_10 = HERE.parent / "sensor_placement" / "sensor_data_10s.json"
BASE_CONFIG = HERE / "BayesianParameters.json"
MAIN = HERE / "MainBayesian.py"

SIGMA = 3.788e-08
SENSOR_COUNTS = [1, 4, 10]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_dataset(dataset):
    """Collapsed means, valid-specimen alphas and u_ref from the cleaned Phase 1 run."""
    collapsed_file = dataset / "measured_data_collapsed_clean.csv"
    clean_file = dataset / "phase1_samples_clean.csv"
    for path, how in ((clean_file, "clean_phase1.py"), (collapsed_file, "collapse_phase1.py")):
        if not path.exists():
            sys.exit(f"missing {path}\nrun {how} on {dataset} first")

    collapsed = read_csv(collapsed_file)
    valid = [r for r in read_csv(clean_file) if r["valid"].strip() == "1"]
    names = [r["name"] for r in collapsed]
    n_collapsed = {int(r["n_samples"]) for r in collapsed}
    if n_collapsed != {len(valid)}:
        sys.exit(f"collapsed file used {n_collapsed} specimens but the clean file has "
                 f"{len(valid)} valid ones -- re-run collapse_phase1.py")

    alpha = np.array([float(r["alpha_true"]) for r in valid])
    E = np.array([float(r["E_true"]) for r in valid])
    u_ref = np.array([np.median(alpha * np.array([float(r[f"u_true_{s}"]) for r in valid]))
                      for s in names])
    return {
        "names": names,
        "rows": collapsed,
        "fields": list(collapsed[0]),
        "u_true_mean": np.array([float(r["u_mean"]) for r in collapsed]),
        "u_hat_mean": np.array([float(r["value"]) for r in collapsed]),
        "u_ref": u_ref,
        "n_valid": len(valid),
        "alpha_pop_mean": float(alpha.mean()),
        "alpha_harmonic_mean": float(1.0 / np.mean(1.0 / alpha)),
        "alpha_sd": float(alpha.std(ddof=1)),
        "e_ref": float(np.median(E / alpha)),
    }


def rebase(path, dest):
    """'output/vtk_output' -> <dest>/vtk_output;  'output' -> <dest>."""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    return str(dest.joinpath(*parts[1:])) if len(parts) > 1 else str(dest)


def write_inputs(dest, k, tag, sensors, data, base_config):
    dest.mkdir(parents=True, exist_ok=True)

    sensor_file = dest / "sensor_data.json"
    sensor_file.write_text(json.dumps({"list_of_sensors": sensors[:k]}, indent=4))

    measured_file = dest / "measured_data.csv"
    with open(measured_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data["fields"])
        writer.writeheader()
        writer.writerows(data["rows"][:k])

    cfg = json.loads(json.dumps(base_config))
    cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_u_mean_{tag}"
    fm = cfg["forward_model"]
    fm["primal_parameters_file"] = str(HERE / fm["primal_parameters_file"])
    fm["sensor_data_file"] = str(sensor_file)
    cfg["likelihood"]["measured_data_file"] = str(measured_file)
    cfg["likelihood"]["noise_model"]["sigma"] = SIGMA
    if "batch_inference" in cfg:
        cfg["batch_inference"]["enabled"] = False
    for proc in cfg["output_processes"]:
        if "output_path" in proc["Parameters"]:
            proc["Parameters"]["output_path"] = rebase(proc["Parameters"]["output_path"], dest)

    config_file = dest / "BayesianParameters.json"
    config_file.write_text(json.dumps(cfg, indent=4))
    return config_file, cfg


def run_phase2(config_file, log_file):
    """MainBayesian.py from bayesian_inference/ (its primal file uses relative paths)."""
    with open(log_file, "w") as log:
        proc = subprocess.Popen([sys.executable, str(MAIN), str(config_file)], cwd=HERE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description="mean-only sensor-count sweep")
    ap.add_argument("tags", nargs="*", help="cases to run, e.g. N04 (default: all)")
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the per-case inputs and run_info.json, skip Phase 2")
    args = ap.parse_args()

    cases = {f"N{k:02d}": k for k in SENSOR_COUNTS}
    unknown = set(args.tags) - set(cases)
    if unknown:
        sys.exit(f"unknown tag(s): {', '.join(sorted(unknown))}\n"
                 f"available: {', '.join(cases)}")
    wanted = args.tags or list(cases)

    data = load_dataset(args.dataset)
    sensors = json.loads(SENSORS_10.read_text())["list_of_sensors"]
    if [s["name"] for s in sensors] != data["names"]:
        sys.exit(f"{SENSORS_10} order {[s['name'] for s in sensors]} does not match the "
                 f"collapsed rows {data['names']}")
    base_config = json.loads(BASE_CONFIG.read_text())
    prior = base_config["parameters"][0]["prior"]

    print(f"dataset      : {args.dataset}  ({data['n_valid']} valid specimens)")
    print(f"truth        : alpha pop mean {data['alpha_pop_mean']:.6f}   "
          f"harmonic mean {data['alpha_harmonic_mean']:.6f}")
    print(f"sigma        : {SIGMA:.4e} for every sensor (option A)")
    print(f"archive      : {args.archive}")

    results = []
    for tag in wanted:
        k = cases[tag]
        dest = args.archive / tag
        u_true = data["u_true_mean"][:k]
        n_eff = float(np.sum(u_true ** 2) / u_true[0] ** 2)
        u_ref, u_hat = data["u_ref"][:k], data["u_hat_mean"][:k]
        # least squares on 1/alpha for u_hat = u_ref / alpha: where the posterior should sit
        alpha_ls = float(np.sum(u_ref ** 2) / np.sum(u_ref * u_hat))
        print(f"\n{'=' * 70}\n  {tag}: {k} sensor{'s' if k > 1 else ''}   N_eff = {n_eff:.4f}"
              f"   -> {dest}\n{'=' * 70}", flush=True)

        config_file, _ = write_inputs(dest, k, tag, sensors, data, base_config)
        started = datetime.now()
        code = None
        if not args.prepare_only:
            code = run_phase2(config_file, dest / "phase2.log")
            if code != 0:
                print(f"\n  *** Phase 2 FAILED (exit {code}) for {tag} ***", flush=True)
            elif not (dest / "summary.json").exists():
                print(f"\n  *** Phase 2 exited cleanly but wrote no summary.json in {dest} ***",
                      flush=True)
        elapsed = (datetime.now() - started).total_seconds()

        (dest / "run_info.json").write_text(json.dumps({
            "label": tag,
            "n_sensors": k,
            "sensor_names": data["names"][:k],
            "stations": [s["location"][0] for s in sensors[:k]],
            "sigma_assumed": SIGMA,
            "sigma_mode": "option A: one instrument sigma for every sensor",
            "dataset": str(args.dataset),
            "n_valid_specimens": data["n_valid"],
            "u_true_mean": u_true.tolist(),
            "u_hat_mean": u_hat.tolist(),
            "u_ref": u_ref.tolist(),
            "n_eff": n_eff,
            "alpha_pop_mean": data["alpha_pop_mean"],
            "alpha_harmonic_mean": data["alpha_harmonic_mean"],
            "alpha_population_sd": data["alpha_sd"],
            "alpha_ls_from_data": alpha_ls,
            "e_ref": data["e_ref"],
            "prior_type": prior["type"],
            "prior_parameters": prior["parameters"],
            "config": str(config_file),
            "returncode": code,
            "wall_time_s": None if args.prepare_only else round(elapsed, 1),
            "started": started.isoformat(timespec="seconds"),
        }, indent=2))
        results.append((tag, code, elapsed))

    print(f"\n{'prepared' if args.prepare_only else 'ran'} {len(results)} case(s)")
    for tag, code, elapsed in results:
        state = "prepared" if code is None else ("ok" if code == 0 else f"FAILED ({code})")
        print(f"  {tag:<6} {state:<12} {elapsed:>7.0f} s")
    print(f"archive: {args.archive}")


if __name__ == "__main__":
    main()
