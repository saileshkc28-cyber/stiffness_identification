"""Gauss-Hermite population check, one zone at a time.

The population is known: each zone's alpha ~ N(true alpha, alpha_std). Each zone is
represented by the three Gauss-Hermite points mu - sqrt(3) sd, mu, mu + sqrt(3) sd with
weights 1/6, 2/3, 1/6, the other zones held at their mean. The centre point is shared,
so three zones need 1 + 2 * 3 = 7 inversions. Readings are noise free, as in the
original one-zone solver; the likelihood uses the sensor noise.

This validates that the inversion returns the population it is given. It does not
infer an unknown population. Run from the case's bayesian_inference folder.
"""
import copy
import gc
import json
import os

import numpy as np
import KratosMultiphysics as Kratos
from kratos_forward_model import KratosForwardModel
from bayesian_analysis import BayesianAnalysis

OUTPUT = os.path.join("output", "gauss_hermite")
Z = np.sqrt(3.0)
W_CENTRE, W_SIDE = 2.0 / 3.0, 1.0 / 6.0

with open("BayesianParameters.json") as f:
    base = json.load(f)
with open("../case.json") as f:
    alpha_std = json.load(f)["population"]["alpha_std"]
with open("../noise_sigma.json") as f:
    sigma = json.load(f)["sigma"]
with open("../damaged_system/StructuralMaterials.json") as f:
    blocks = json.load(f)["properties"]
with open(base["forward_model"]["sensor_data_file"]) as f:
    sensors = json.load(f)["list_of_sensors"]


def write_readings(path, values):
    with open(path, "w") as f:
        f.write("#,type,name,location_0,location_1,location_2,value\n")
        for i, (s, v) in enumerate(zip(sensors, values), 1):
            f.write(f"{i},{s['type']},{s['name']},{s['location'][0]},"
                    f"{s['location'][1]},{s['location'][2]},{v:.16e}\n")


# 1. Points and noise-free readings, from one forward model
settings = Kratos.Parameters(json.dumps(base))
entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
fm = KratosForwardModel(Kratos.Model(), settings["forward_model"], entries)
truth = {b["properties_id"]: b["Material"]["Variables"]["YOUNG_MODULUS"]
         for b in blocks if "YOUNG_MODULUS" in b["Material"]["Variables"]}
mu = np.array([truth[ids[0]] / ref for ids, ref in zip(fm.prop_ids, fm.refs)])
n_zones = len(mu)

points = [("centre", None, 0.0, mu.copy())]
for zone in range(n_zones):
    for side, z in (("low", -Z), ("high", Z)):
        alpha = mu.copy()
        alpha[zone] += z * alpha_std
        if alpha[zone] <= 0.0:
            raise RuntimeError(f"zone {zone + 1} {side} point is {alpha[zone]:.3f}; "
                               "lower alpha_std")
        points.append((f"zone{zone + 1}_{side}", zone, z, alpha))

os.makedirs(OUTPUT, exist_ok=True)
for i, (name, _, _, alpha) in enumerate(points, 1):
    folder = os.path.join(OUTPUT, f"point_{i:02d}_{name}")
    os.makedirs(folder, exist_ok=True)
    write_readings(os.path.join(folder, "measured_data.csv"), fm.Evaluate(alpha))
data_solves = fm.n_solves
fm.Finalize()
del fm
gc.collect()

# 2. One ordinary inversion per point, each writing into its own folder
means, sds, solves = [], [], 0
for i, (name, _, _, alpha) in enumerate(points, 1):
    folder = os.path.join(OUTPUT, f"point_{i:02d}_{name}")
    run = copy.deepcopy(base)
    run["problem_data"]["problem_name"] = f"gauss_hermite_{name}"
    run["likelihood"]["measured_data_file"] = os.path.join(folder, "measured_data.csv")
    run["likelihood"]["noise_model"]["sigma"] = sigma
    for process in run["output_processes"]:
        path = process["Parameters"].get("output_path", "output")
        process["Parameters"]["output_path"] = os.path.join(
            folder, os.path.relpath(path, "output"))
    print(f"\n=== point {i} of {len(points)}: {name}, true alpha "
          f"{np.array2string(alpha, precision=4)}")
    BayesianAnalysis(Kratos.Model(), Kratos.Parameters(json.dumps(run))).Run()
    gc.collect()

    with open(os.path.join(folder, "summary.json")) as f:
        result = json.load(f)
    means.append([z["alpha_mean"] for z in result["zones"]])
    sds.append([z["alpha_std"] for z in result["zones"]])
    solves += result["n_forward_solves"]
means, sds = np.array(means), np.array(sds)

# 3. Combine per zone: the centre point and that zone's own low and high points
zones = []
for zone in range(n_zones):
    rows = [0, 1 + 2 * zone, 2 + 2 * zone]
    weights = np.array([W_CENTRE, W_SIDE, W_SIDE])
    m = means[rows, zone]
    mean = float(weights @ m)
    sd = float(np.sqrt(weights @ (m - mean) ** 2))
    # moving one zone should leave the others where they are
    others = [r for r in range(1, len(points)) if points[r][1] != zone]
    leak = float(np.abs(means[others, zone] - mu[zone]).max())
    zones.append({"zone": zone + 1, "target_mean": float(mu[zone]),
                  "target_sd": alpha_std, "recovered_mean": mean,
                  "recovered_sd": sd, "max_shift_when_other_zones_move": leak})

summary = {"method": "Gauss-Hermite, one zone at a time, 3 points per zone, shared centre",
           "alpha_std": alpha_std, "likelihood_sigma": sigma,
           "measurement": "noise free",
           "points": [{"name": p[0], "alpha_true": p[3].tolist(),
                       "posterior_mean": means[i].tolist(),
                       "posterior_sd": sds[i].tolist()}
                      for i, p in enumerate(points)],
           "zones": zones, "n_inversions": len(points),
           "forward_solves_data": data_solves, "forward_solves_inference": solves}
with open(os.path.join(OUTPUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nGauss-Hermite: {len(points)} inversions, {solves} inference solves")
for z in zones:
    print(f"zone {z['zone']}: target {z['target_mean']:.4f} +/- {z['target_sd']:.4f}   "
          f"recovered {z['recovered_mean']:.4f} +/- {z['recovered_sd']:.4f}   "
          f"shift when other zones move {z['max_shift_when_other_zones_move']:.4f}")
print(f"written: {os.path.join(OUTPUT, 'summary.json')}")
