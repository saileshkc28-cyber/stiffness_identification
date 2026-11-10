"""Build the additive surrogate for a case, then check it.

1. Base run: every zone at the centre ("anchor", e.g. the mean-only result; never the
   truth).
2. One curve per zone: change that zone alone over anchor +/- half_width, the others
   stay at the anchor, and fit the change in every sensor reading with the
   one-dimensional GP. The runs are spaced evenly in log(alpha) on each side of the
   anchor, so the steep soft end gets more of them and the anchor itself is a run.
3. Optional pair corrections ("pairs": "neighbours"): for every two zones that share
   mesh nodes, runs changing both together on a coarser copy of the curve grid, and a
   two-dimensional GP of what the two single curves miss. The single-zone changes are
   taken from the exact curve runs at the same alpha values, not from the fitted
   curves, so a zone that belongs to two pairs is not corrected twice.
4. Check: full Kratos runs in which every zone changes at once, compared with the
   surrogate. The error is measured in units of the sensor noise sigma and must stay
   below the gate, as for the one-zone surrogate.

Run from the case's bayesian_inference folder. Settings come from the "surrogate" block
of ../case.json.
"""
import csv
import json
import os
import sys

import joblib
import numpy as np
import KratosMultiphysics as Kratos
from kratos_forward_model import KratosForwardModel
from response_surrogate import ResponseSurrogate
from additive_surrogate import AdditiveSurrogate, PairCurve

OUTPUT = os.path.join("output", "surrogate")

with open("BayesianParameters.json") as f:
    settings = Kratos.Parameters(f.read())
with open("../case.json") as f:
    cfg = json.load(f)["surrogate"]
with open("../noise_sigma.json") as f:
    sigma = json.load(f)["sigma"]
with open(settings["forward_model"]["sensor_data_file"].GetString()) as f:
    sensor_names = [s["name"] for s in json.load(f)["list_of_sensors"]]

entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
zone_names = [e["name"].GetString() for e in entries]
fm = KratosForwardModel(Kratos.Model(), settings["forward_model"], entries)
n_zones = len(entries)
e_refs = np.array(fm.refs, float)

anchor = np.array(cfg["anchor"], float)
if anchor.size != n_zones:
    raise RuntimeError(f"anchor has {anchor.size} values for {n_zones} zones")
lo = np.maximum(anchor - cfg["half_width"], cfg["alpha_floor"])
hi = anchor + cfg["half_width"]
half, odd = divmod(cfg["points_per_zone"] - 1, 2)
if odd:
    raise RuntimeError("points_per_zone must be odd, so the anchor is one of the runs")
os.makedirs(OUTPUT, exist_ok=True)

# 1. base run at the anchor
u_base = np.asarray(fm.Evaluate(anchor), float)
training = [["base", float("nan")] + list(u_base)]

# 2. one curve per zone
curves, exact_change = [], []
for z in range(n_zones):
    grid = np.concatenate([
        np.exp(np.linspace(np.log(lo[z]), np.log(anchor[z]), half + 1)),
        np.exp(np.linspace(np.log(anchor[z]), np.log(hi[z]), half + 1))[1:]])
    alphas, changes = [], []
    for a in grid:
        alpha = anchor.copy()
        alpha[z] = a
        u = np.asarray(fm.Evaluate(alpha), float)
        training.append([zone_names[z], a] + list(u))
        alphas.append(a)
        changes.append(u - u_base)
    exact_change.append((grid, np.array(changes)))
    if np.abs(grid - anchor[z]).min() > 1e-9:      # zero change at the anchor
        alphas.append(anchor[z])
        changes.append(np.zeros_like(u_base))
    curve = ResponseSurrogate(e_refs[z], lo[z] * e_refs[z], hi[z] * e_refs[z],
                              gp_jitter=cfg["gp_jitter"], n_restarts=cfg["n_restarts"],
                              sensor_names=sensor_names, random_state=cfg["seed"])
    curves.append(curve.fit(np.array(alphas) * e_refs[z], np.array(changes)))
    print(f"{zone_names[z]}: curve over alpha {lo[z]:.3f} to {hi[z]:.3f}, {len(grid)} runs")
first_order = AdditiveSurrogate(u_base, e_refs, curves, sensor_names, zone_names,
                                 anchor)

# 3. pair corrections for zones that touch, i.e. share mesh nodes
pairs = []
n_pair_runs = 0
if cfg.get("pairs", "none") == "neighbours":
    n_pair = cfg["points_per_pair_axis"]
    step, rest = divmod(cfg["points_per_zone"] - 1, n_pair - 1)
    if rest or half % step:
        raise RuntimeError("points_per_pair_axis does not fit the curve grid: need "
                           "(points_per_zone - 1) divisible by (points_per_pair_axis - 1) "
                           "and (points_per_zone - 1) / 2 divisible by the step, so the "
                           "pair grid sits on the curve grid and includes the anchor")
    nodes = [set(np.asarray(fm.cells)[zone[3]].ravel()) for zone in fm.zones]
    neighbours = [(i, j) for i in range(n_zones) for j in range(i + 1, n_zones)
                  if nodes[i] & nodes[j]]
    for i, j in neighbours:
        grid_i, change_i = exact_change[i]
        grid_j, change_j = exact_change[j]
        points, misses = [], []
        for p in range(0, len(grid_i), step):
            for q in range(0, len(grid_j), step):
                a_i, a_j = grid_i[p], grid_j[q]
                points.append([a_i, a_j])
                if abs(a_i - anchor[i]) < 1e-9 or abs(a_j - anchor[j]) < 1e-9:
                    misses.append(np.zeros_like(u_base))   # zero by definition
                    continue
                alpha = anchor.copy()
                alpha[i], alpha[j] = a_i, a_j
                u = np.asarray(fm.Evaluate(alpha), float)
                n_pair_runs += 1
                training.append([f"{zone_names[i]}+{zone_names[j]}", float("nan")] + list(u))
                misses.append(u - u_base - change_i[p] - change_j[q])
        pair = PairCurve(i, j, [lo[i], lo[j]], [hi[i], hi[j]], gp_jitter=cfg["gp_jitter"],
                         n_restarts=cfg["n_restarts"], random_state=cfg["seed"])
        pairs.append(pair.fit(np.array(points), np.array(misses)))
        print(f"pair {zone_names[i]} + {zone_names[j]}: correction on a "
              f"{n_pair} x {n_pair} grid")
    if not neighbours:
        print("no zones share mesh nodes: no pair corrections")
surrogate = AdditiveSurrogate(u_base, e_refs, curves, sensor_names, zone_names,
                               anchor, pairs)

# 4. check against full runs with every zone changed at once
rng = np.random.default_rng(cfg["seed"])
check = rng.uniform(lo, hi, (cfg["n_check"], n_zones))
u_full = np.array([fm.Evaluate(a) for a in check], float)
n_solves = fm.n_solves
fm.Finalize()

u_pred = surrogate.predict(check)
err = np.abs(u_pred - u_full) / sigma            # (n_check, n_sensors)
err_first = float((np.abs(first_order.predict(check) - u_full) / sigma).max())
worst = float(err.max())
gate = cfg["max_error_over_sigma"]
passed = worst < gate
i_worst = int(err.max(axis=1).argmax())

with open(os.path.join(OUTPUT, "training_runs.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["zone_changed", "alpha"] + [f"u_{n}" for n in sensor_names])
    writer.writerows([[r[0]] + [repr(float(v)) for v in r[1:]] for r in training])
with open(os.path.join(OUTPUT, "check_runs.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"alpha_{z + 1}" for z in range(n_zones)]
                    + [f"u_{n}" for n in sensor_names]
                    + [f"error_over_sigma_{n}" for n in sensor_names])
    for a, u, e in zip(check, u_full, err):
        writer.writerow([repr(float(v)) for v in list(a) + list(u) + list(e)])
joblib.dump(surrogate, os.path.join(OUTPUT, "surrogate.joblib"))

report = {"method": "additive: one curve per zone, GP per sensor"
                    + (", plus pair corrections for neighbouring zones" if pairs else ""),
          "pairs": [[p.i + 1, p.j + 1] for p in pairs],
          "max_error_over_sigma_curves_only": err_first,
          "anchor": anchor.tolist(), "alpha_low": lo.tolist(), "alpha_high": hi.tolist(),
          "points_per_zone": cfg["points_per_zone"],
          "n_check": cfg["n_check"], "sigma": sigma, "gate_error_over_sigma": gate,
          "max_error_over_sigma": worst, "passed": passed,
          "per_sensor_max_error_over_sigma": err.max(axis=0).tolist(),
          "worst_check_alpha": check[i_worst].tolist(),
          "kratos_solves": n_solves,
          "kernels": [[str(g.kernel_) for g in c.gps_] for c in curves]}
with open(os.path.join(OUTPUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{n_solves} Kratos solves: 1 base, {n_zones} x {cfg['points_per_zone']} curve runs, "
      f"{n_pair_runs} pair runs, {cfg['n_check']} check runs")
print(f"curves only: worst {err_first:.4f} sigma")
print("check, max error per sensor in sigma:",
      np.array2string(err.max(axis=0), precision=4))
print(f"worst {worst:.4f} sigma at alpha {np.array2string(check[i_worst], precision=3)}"
      f"   gate {gate}   {'PASS' if passed else 'FAIL'}")
print(f"written: {OUTPUT}")
if not passed:
    sys.exit("surrogate failed the check: do not use it for the hierarchical run")
