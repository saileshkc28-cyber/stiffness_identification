r"""Weighted three-point sensor-count sweep. Edits nothing in the repository.

Phase 2 runs the three-point Gauss-Hermite study (alpha = 1 -+ sqrt3 * 0.1 and 1,
weights 1/6, 2/3, 1/6) on the first k sensors of the 10-sensor layout (tip first,
then inward), for k in 1, 4, 10. Every input a run needs is written into its own
output folder, and the run writes everything else there too:

    <archive>/N04/sensor_data.json          first 4 sensors of sensor_data_10s.json
    <archive>/N04/BayesianParameters.json   BayesianParameters.json pointed at the file
                                            above, sigma 3.788e-08, three-point on,
                                            batch off, input and output paths in N04/
    <archive>/N04/run_info.json             k, stations, sigma, target, alpha points,
                                            weights, prior, N_eff and u_true of the
                                            central point
    <archive>/N04/phase2.log                MainBayesian.py console output
    <archive>/N04/three_point_inputs/       the three generated measurements
    <archive>/N04/point_1_low/ ... combined/  the three inversions and their combination

The output processes keep their repo-relative paths ('output', 'output/vtk_output'):
in three-point mode run_case re-roots them under <archive>/N04/point_*/, so nothing
is written into the repository. An absolute path there would be nested inside each
point folder instead.

The repo's sensor_data_10s.json and BayesianParameters.json are only read.

Noise model, option A: one instrument sigma = 3.788e-08 for every sensor, as in the
sensor, u_mean and hierarchical sweeps. Measurements use zero_mean_noise, so the
three u_hat are the clean Kratos responses.

Truth: the prescribed Phase 1 population alpha ~ N(1.0, 0.1), read from
three_point_inference.alpha_mean / alpha_std in the config. The three points are
built from it, and each point's own truth is its alpha_j.

Run from bayesian_inference/:
    python run_weighted_sweep.py                 # N01, N04, N10
    python run_weighted_sweep.py N04             # re-run single cases
    python run_weighted_sweep.py --prepare-only  # write inputs, skip MainBayesian.py
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\KratosProjects\MCMC\Analysis\sensors_weighted")
SENSORS_10 = HERE.parent / "sensor_placement" / "sensor_data_10s.json"
BASE_CONFIG = HERE / "BayesianParameters.json"
MAIN = HERE / "MainBayesian.py"

SIGMA = 3.788e-08
SENSOR_COUNTS = [1, 4, 10]

# the rule in three_point_bayesian_analysis.py, repeated so this script needs no Kratos
Z_VALUES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
WEIGHTS = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def is_fresh(path, started):
    """Exists and was written by this run, not left over from an earlier one."""
    return path.exists() and path.stat().st_mtime >= started.timestamp() - 1.0


def check_output_paths(cfg):
    """run_case re-roots these per point; that only works for relative paths."""
    for proc in cfg["output_processes"]:
        path = proc["Parameters"].get("output_path")
        if path is not None and (os.path.isabs(path) or Path(path).drive):
            sys.exit(f"output process {proc['python_module']} has an absolute output_path "
                     f"{path!r}; three-point mode needs a relative one")


def write_inputs(dest, k, tag, sensors, base_config):
    dest.mkdir(parents=True, exist_ok=True)

    sensor_file = dest / "sensor_data.json"
    sensor_file.write_text(json.dumps({"list_of_sensors": sensors[:k]}, indent=4))

    cfg = json.loads(json.dumps(base_config))
    cfg["problem_data"]["problem_name"] = f"bayesian_inference_beam_weighted_{tag}"
    fm = cfg["forward_model"]
    fm["primal_parameters_file"] = str(HERE / fm["primal_parameters_file"])
    fm["sensor_data_file"] = str(sensor_file)
    cfg["likelihood"]["noise_model"]["sigma"] = SIGMA
    if "batch_inference" in cfg:
        cfg["batch_inference"]["enabled"] = False
    tp = cfg["three_point_inference"]
    tp["enabled"] = True
    tp["measurement_noise_sigma"] = SIGMA
    tp["input_path"] = str(dest / "three_point_inputs")
    tp["output_path"] = str(dest)
    tp["resume"] = False
    tp["dry_run"] = False

    config_file = dest / "BayesianParameters.json"
    config_file.write_text(json.dumps(cfg, indent=4))
    return config_file, cfg


def run_phase2(config_file, log_file):
    """MainBayesian.py from bayesian_inference/ (its primal file uses relative paths)."""
    # unbuffered child so its prints reach the screen and the log as they happen
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_file, "w") as log:
        proc = subprocess.Popen([sys.executable, str(MAIN), str(config_file)], cwd=HERE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def central_response(dest, started):
    """u_true of the central point and N_eff = sum u^2 / u_tip^2, or (None, None)."""
    meta = dest / "three_point_inputs" / "point_2_central" / "metadata.json"
    if not is_fresh(meta, started):
        return None, None
    u = np.array(json.loads(meta.read_text())["u_true"], dtype=float)
    return u.tolist(), float(np.sum(u ** 2) / u[0] ** 2)


def main():
    ap = argparse.ArgumentParser(description="weighted three-point sensor-count sweep")
    ap.add_argument("tags", nargs="*", help="cases to run, e.g. N04 (default: all)")
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the per-case inputs and run_info.json, skip Phase 2")
    args = ap.parse_args()

    cases = {f"N{k:02d}": k for k in SENSOR_COUNTS}
    unknown = set(args.tags) - set(cases)
    if unknown:
        sys.exit(f"unknown tag(s): {', '.join(sorted(unknown))}\n"
                 f"available: {', '.join(cases)}")
    wanted = args.tags or list(cases)

    sensors = json.loads(SENSORS_10.read_text())["list_of_sensors"]
    if len(sensors) < max(SENSOR_COUNTS):
        sys.exit(f"{SENSORS_10} has {len(sensors)} sensors, need {max(SENSOR_COUNTS)}")
    base_config = json.loads(BASE_CONFIG.read_text())
    if "three_point_inference" not in base_config:
        sys.exit(f"{BASE_CONFIG} has no three_point_inference block")
    check_output_paths(base_config)
    prior = base_config["parameters"][0]["prior"]
    tp = base_config["three_point_inference"]
    alpha_mean, alpha_sd, e_ref = tp["alpha_mean"], tp["alpha_std"], tp["E_ref"]
    alpha_points = alpha_mean + Z_VALUES * alpha_sd

    print(f"sensors      : {SENSORS_10}  (first k of {len(sensors)})")
    print(f"truth        : alpha ~ N({alpha_mean}, {alpha_sd})   "
          f"points {', '.join(f'{a:.6f}' for a in alpha_points)}   "
          f"weights {', '.join(f'{w:.4f}' for w in WEIGHTS)}")
    print(f"sigma        : {SIGMA:.4e} for every sensor (option A)")
    print(f"archive      : {args.archive}")

    results = []
    for tag in wanted:
        k = cases[tag]
        dest = args.archive / tag
        stations = [s["location"][0] for s in sensors[:k]]
        print(f"\n{'=' * 70}\n  {tag}: {k} sensor{'s' if k > 1 else ''}   "
              f"x = {', '.join(format(x, '.1f') for x in stations)}"
              f"   -> {dest}\n{'=' * 70}", flush=True)

        config_file, _ = write_inputs(dest, k, tag, sensors, base_config)
        started = datetime.now()
        code = None
        if not args.prepare_only:
            code = run_phase2(config_file, dest / "phase2.log")
            if code != 0:
                print(f"\n  *** Phase 2 FAILED (exit {code}) for {tag} ***", flush=True)
            elif not is_fresh(dest / "combined" / "combined_summary.json", started):
                print(f"\n  *** Phase 2 exited cleanly but wrote no "
                      f"combined/combined_summary.json in {dest} ***", flush=True)
        elapsed = (datetime.now() - started).total_seconds()
        u_true, n_eff = (None, None) if args.prepare_only else central_response(dest, started)

        (dest / "run_info.json").write_text(json.dumps({
            "label": tag,
            "n_sensors": k,
            "sensor_names": [s["name"] for s in sensors[:k]],
            "stations": stations,
            "sigma_assumed": SIGMA,
            "sigma_mode": "option A: one instrument sigma for every sensor",
            "alpha_target_mean": alpha_mean,
            "alpha_target_sd": alpha_sd,
            "alpha_points": alpha_points.tolist(),
            "z_values": Z_VALUES.tolist(),
            "weights": WEIGHTS.tolist(),
            "e_ref": e_ref,
            "u_true_central": u_true,
            "n_eff": n_eff,
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
