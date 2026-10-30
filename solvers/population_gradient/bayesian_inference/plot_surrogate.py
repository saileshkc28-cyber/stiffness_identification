"""Plot the response surrogate: fitted curve, training and validation points,
and the validation residuals against the accuracy gate.

    python plot_surrogate.py [SurrogateParameters.json]

Reads the solves already on file; runs nothing.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from response_surrogate import ResponseSurrogate


def to_float(x):
    """numpy 2 reprs as 'np.float64(1.0)'; accept those as well as plain text."""
    if isinstance(x, str) and x.startswith("np.float"):
        x = x[x.index("(") + 1:x.rindex(")")]
    return float(x)


def read_points(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    e = np.array([to_float(r["E_Pa"]) for r in rows])
    cols = sorted(c for c in rows[0] if c.startswith("u_"))
    u = np.array([[to_float(r[c]) for c in cols] for r in rows])
    order = np.argsort(e)
    return e[order], u[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="SurrogateParameters.json")
    ap.add_argument("--sensor", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out = cfg["output"]

    gp = ResponseSurrogate.load(out["model_file"])
    e_tr, u_tr = read_points(out["training_csv"])
    e_va, u_va = read_points(out["validation_csv"])

    sigma = np.asarray(cfg["accuracy_gate"]["sigma_noise_m"], float).ravel()
    gate = cfg["accuracy_gate"]["max_error_over_sigma"] * sigma[args.sensor]

    e_fine = np.exp(np.linspace(np.log(gp.e_min), np.log(gp.e_max), 600))
    mean, sd = gp.predict(e_fine, return_std=True)
    k = args.sensor

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.fill_between(e_fine / 1e9, (mean[:, k] - 2 * sd[:, k]) * 1e6,
                    (mean[:, k] + 2 * sd[:, k]) * 1e6,
                    color="0.85", label="GP 95% band")
    ax.plot(e_fine / 1e9, mean[:, k] * 1e6, "-", color="C0", lw=1.6,
            label="GP surrogate")
    ax.plot(e_tr / 1e9, u_tr[:, k] * 1e6, "o", ms=5, color="C3",
            label=f"training solves (n={e_tr.size})")
    ax.plot(e_va / 1e9, u_va[:, k] * 1e6, "x", ms=7, mew=1.6, color="C2",
            label=f"validation solves (n={e_va.size})")
    ax.set_xscale("log")
    ax.set_xlabel("Young's modulus  $E$  [GPa]")
    ax.set_ylabel(r"tip deflection  $u$  [$\mu$m]")
    ax.set_title("Response surrogate  $u(E)$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    res_tr = gp.predict(e_tr)[:, k] - u_tr[:, k]
    res_va = gp.predict(e_va)[:, k] - u_va[:, k]

    fig.tight_layout()
    png = os.path.join(out["dir"], "surrogate_fit.png")
    fig.savefig(png, dpi=160)
    print(f"max |training residual|   {np.abs(res_tr).max():.3e} m")
    print(f"max |validation residual| {np.abs(res_va).max():.3e} m  "
          f"gate {gate:.3e} m  -> {'PASS' if np.abs(res_va).max() < gate else 'FAIL'}")
    print(f"-> {png}")


if __name__ == "__main__":
    main()