"""Population Phase 1: many specimens, each zone's alpha drawn from a normal
distribution around that zone's true value. Writes every specimen, and the averaged
readings that the mean-only inference uses. Run from the case's bayesian_inference
folder."""
import csv
import json

import numpy as np
import KratosMultiphysics as Kratos
from kratos_forward_model import KratosForwardModel

with open("BayesianParameters.json") as f:
    settings = Kratos.Parameters(f.read())
with open("../case.json") as f:
    population = json.load(f)["population"]
with open("../noise_sigma.json") as f:
    sigma = json.load(f)["sigma"]
with open("../damaged_system/StructuralMaterials.json") as f:
    blocks = json.load(f)["properties"]
with open(settings["forward_model"]["sensor_data_file"].GetString()) as f:
    sensors = json.load(f)["list_of_sensors"]

entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
model = Kratos.Model()
fm = KratosForwardModel(model, settings["forward_model"], entries)
n_zones = len(entries)

# The population of each zone is centred on that zone's true alpha.
truth = {b["properties_id"]: b["Material"]["Variables"]["YOUNG_MODULUS"]
         for b in blocks if "YOUNG_MODULUS" in b["Material"]["Variables"]}
alpha_mean = np.array([truth[ids[0]] / ref for ids, ref in zip(fm.prop_ids, fm.refs)])
alpha_std = population["alpha_std"]
n_samples = population["n_samples"]

rng = np.random.default_rng(population["seed"])
alpha = alpha_mean + alpha_std * rng.standard_normal((n_samples, n_zones))
if alpha.min() <= 0.0:
    raise RuntimeError(f"a drawn alpha is {alpha.min():.3f}; lower alpha_std")

u_true = np.array([fm.Evaluate(a) for a in alpha])
u_hat = u_true + rng.normal(0.0, sigma, u_true.shape)
n_solves = fm.n_solves
fm.Finalize()

names = [s["name"] for s in sensors]
with open("../damaged_system/population_samples.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id"] + [f"alpha_{z + 1}" for z in range(n_zones)]
                    + [f"u_true_{n}" for n in names] + [f"u_hat_{n}" for n in names])
    for i in range(n_samples):
        writer.writerow([i + 1] + [repr(float(v)) for v in alpha[i]]
                        + [repr(float(v)) for v in u_true[i]]
                        + [repr(float(v)) for v in u_hat[i]])

# The mean-only inference reads this file, in the same format as Phase 1.
u_mean = u_hat.mean(axis=0)
with open("../damaged_system/measured_data.csv", "w") as f:
    f.write("#,type,name,location_0,location_1,location_2,value\n")
    for i, (s, v) in enumerate(zip(sensors, u_mean), 1):
        f.write(f"{i},{s['type']},{s['name']},{s['location'][0]},{s['location'][1]},"
                f"{s['location'][2]},{v:.16e}\n")

# Readings average 1/alpha, not alpha, so mean-only should land near the harmonic mean.
harmonic = 1.0 / np.mean(1.0 / alpha, axis=0)
summary = {"n_samples": n_samples, "seed": population["seed"], "sigma": sigma,
           "n_forward_solves": n_solves,
           "alpha_target_mean": alpha_mean.tolist(), "alpha_target_std": alpha_std,
           "alpha_realised_mean": alpha.mean(axis=0).tolist(),
           "alpha_realised_std": alpha.std(axis=0, ddof=1).tolist(),
           "alpha_harmonic_mean": harmonic.tolist()}
with open("../damaged_system/population_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{n_samples} specimens, {n_solves} forward solves, noise sigma {sigma:.6e} m")
for z in range(n_zones):
    print(f"zone {z + 1}: target {alpha_mean[z]:.4f} +/- {alpha_std:.4f}, "
          f"drawn {alpha[:, z].mean():.4f} +/- {alpha[:, z].std(ddof=1):.4f}, "
          f"harmonic mean {harmonic[z]:.4f}")
print("averaged readings:", np.array2string(u_mean, precision=4))
print("written: population_samples.csv, measured_data.csv, population_summary.json "
      "in damaged_system/")
