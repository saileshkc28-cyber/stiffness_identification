"""Forward checks for a case. Run from the case's bayesian_inference folder."""
import json

import numpy as np
import KratosMultiphysics as Kratos
from kratos_forward_model import KratosForwardModel

with open("BayesianParameters.json") as f:
    settings = Kratos.Parameters(f.read())
with open("../case.json") as f:
    case = json.load(f)
with open("../damaged_system/StructuralMaterials.json") as f:
    blocks = json.load(f)["properties"]

entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
model = Kratos.Model()
fm = KratosForwardModel(model, settings["forward_model"], entries)
root = fm.root
n_zones = len(entries)

print(f"\nnodes {root.NumberOfNodes()}, elements {root.NumberOfElements()}, "
      f"conditions {root.NumberOfConditions()}")
for entry, ref, ids in zip(entries, fm.refs, fm.prop_ids):
    name = entry["target_sub_model_part"].GetString()
    print(f"{name}: {model[name].NumberOfElements()} elements, property {ids}, "
          f"E_ref {ref:.4e}")

base = fm.Evaluate(np.ones(n_zones))
reaction_x = sum(n.GetSolutionStepValue(Kratos.REACTION_X) for n in root.Nodes)
reaction_y = sum(n.GetSolutionStepValue(Kratos.REACTION_Y) for n in root.Nodes)
peak = int(np.abs(base).argmax())
print(f"\nreaction x = {reaction_x:.4f} N, y = {reaction_y:.4f} N")
print("undamaged sensors:", np.array2string(base, precision=4))
print(f"largest undamaged reading = {base[peak]:.6e} m (sensor {peak + 1})")

# Halving one zone shows how much of the largest reading comes from that zone.
extra = []
for zone in range(n_zones):
    alpha = np.ones(n_zones)
    alpha[zone] = 0.5
    extra.append(fm.Evaluate(alpha)[peak] - base[peak])
print("share by zone:", np.round(np.array(extra) / sum(extra), 4))

truth = {b["properties_id"]: b["Material"]["Variables"]["YOUNG_MODULUS"]
         for b in blocks if "YOUNG_MODULUS" in b["Material"]["Variables"]}
alpha_true = np.array([truth[ids[0]] / ref for ids, ref in zip(fm.prop_ids, fm.refs)])
print("true alpha:", np.round(alpha_true, 4))
print(f"reading at true alpha = {fm.Evaluate(alpha_true)[peak]:.6e} m")

# A linear model has alpha * u constant. A spurious solve breaks this.
scaling = max(np.abs(a * fm.Evaluate(np.full(n_zones, a)) - base).max()
              for a in (0.2, 2.0)) / abs(base[peak])
repeat = np.abs(fm.Evaluate(np.ones(n_zones)) - base).max()
print(f"scaling error = {scaling:.3e}")
print(f"repeat difference = {repeat:.3e} m")

if "noise_sigma" in case:
    sigma = case["noise_sigma"]
else:
    sigma = case["noise_percent"] / 100.0 * abs(base[peak])
with open("../noise_sigma.json", "w") as f:
    json.dump({"sigma": sigma}, f, indent=2)
print(f"noise sigma = {sigma:.6e} m (written to noise_sigma.json)")

fm.Finalize()
