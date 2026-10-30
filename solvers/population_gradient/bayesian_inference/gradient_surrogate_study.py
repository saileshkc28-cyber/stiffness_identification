"""How many training solves does u(E) need if the slope du/dt is used as well?

Reuses the training and validation solves already on file (no new solves, no
Kratos) and compares, for growing equally spaced training subsets:
    a. ResponseSurrogate  -- the existing GP, values only
    b. InterpBaseline     -- the existing PCHIP, values only
    c. cubic Hermite      -- values u plus slopes du/dt = -u, per sensor in t
    d. GradientEnhancedGP -- Matern 5/2 GP on values u plus slopes du/dt = -u
plus one "analytic" reference row: u(E) = u0 * E0 / E from a single solve, and,
if gradient_training_responses.csv (adjoint_training_solves.py) is in the output
dir, one "adjoint" row: c. and d. fitted with the Kratos adjoint slopes instead.

t = ln(E / e_scale), as in ResponseSurrogate.t_of. For a linear elastic model
u is proportional to 1/E, so du/dt = -u exactly; check 2a tests this on the data.

Usage:
    python gradient_surrogate_study.py
    python gradient_surrogate_study.py --config <path to SurrogateParameters.json>
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import CubicHermiteSpline

# BuildSurrogate.py imports Kratos only inside build_forward_model, so this is safe
from BuildSurrogate import load_done, to_float
from gradient_gp import GradientEnhancedGP
from response_surrogate import InterpBaseline, ResponseSurrogate, error_report

SUBSET_SIZES = (2, 3, 4, 5, 7, 9, 13, 25)
ADJOINT_CSV = "gradient_training_responses.csv"   # written by adjoint_training_solves.py
MODELS = (("gp", "GP"), ("pchip", "PCHIP"), ("hermite", "Hermite"),
          ("gp_slope", "GP+slope"), ("analytic", "analytic"))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_responses(path):
    """Existing solve CSV -> (E sorted ascending, u with one row per E)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"response CSV not found: {os.path.abspath(path)} -- this study only reads "
            "existing solves; run BuildSurrogate.py --stage solves first")
    # load_done needs the sensor count up front; take it from the header
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    n_sensors = sum(c.startswith("u_") for c in header)
    if n_sensors == 0:
        raise RuntimeError(f"{path} has no u_* response columns")
    done = load_done(path, n_sensors)
    if not done:
        raise RuntimeError(f"{path} holds no rows with status 'ok'")
    keys = sorted(done)
    return np.array(keys), np.vstack([done[k] for k in keys])


def load_adjoint(path, n_sensors):
    """Adjoint CSV -> (E sorted ascending, u, dudt), one row per E.
    Every row must have status 'ok'; the ratio_<i> columns are not used here."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        n_u = sum(c.startswith("u_") for c in fields)
        n_d = sum(c.startswith("dudt_") for c in fields)
        if n_u != n_sensors or n_d != n_sensors:
            raise RuntimeError(f"{path} has {n_u} u_* and {n_d} dudt_* column(s), the "
                               f"training CSV has {n_sensors} sensor(s)")
        rows = list(reader)
    bad = [r.get("status") for r in rows if r.get("status") != "ok"]
    if bad:
        raise RuntimeError(f"{path}: {len(bad)} row(s) with status other than 'ok' "
                           f"({', '.join(sorted(set(map(str, bad))))})")
    if not rows:
        raise RuntimeError(f"{path} holds no rows")
    rows.sort(key=lambda r: to_float(r["E_Pa"]))
    e = np.array([to_float(r["E_Pa"]) for r in rows])
    u = np.array([[to_float(r[f"u_{i}"]) for i in range(n_sensors)] for r in rows])
    dudt = np.array([[to_float(r[f"dudt_{i}"]) for i in range(n_sensors)] for r in rows])
    return e, u, dudt


def sensor_labels(model_file, n_sensors):
    """Sensor names from the saved surrogate's identity file if present, else u_i."""
    meta = str(model_file) + ".json"
    if os.path.exists(meta):
        with open(meta) as f:
            names = json.load(f).get("sensor_names")
        if names and len(names) == n_sensors:
            return names
    return [f"u_{i}" for i in range(n_sensors)]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class HermiteInT:
    """Cubic Hermite spline per sensor in t = ln(E / e_scale), slopes du/dt = -u
    unless dudt_train is given."""

    def __init__(self, e_scale_Pa):
        self.e_scale = float(e_scale_Pa)

    def fit(self, e_train_Pa, u_train_m, dudt_train=None):
        t = np.log(np.asarray(e_train_Pa, dtype=float) / self.e_scale)
        u = np.asarray(u_train_m, dtype=float)
        if dudt_train is None:
            # u = C / E = C / (e_scale * exp(t))  =>  du/dt = -u  (natural log)
            dudt = -u
        else:
            # slopes supplied by the caller, e.g. from the Kratos adjoint
            dudt = np.asarray(dudt_train, dtype=float)
            if dudt.shape != u.shape:
                raise ValueError(f"dudt_train shape {dudt.shape} != u shape {u.shape}")
        self.splines_ = [CubicHermiteSpline(t, u[:, s], dudt[:, s])
                         for s in range(u.shape[1])]
        return self

    def predict(self, e_Pa):
        t = np.log(np.asarray(e_Pa, dtype=float).ravel() / self.e_scale)
        return np.column_stack([f(t) for f in self.splines_])


def score(fit_predict, e_va, u_va, sigma, label):
    """max error / sigma for one model; NaN (and a message) if fit or predict fails."""
    try:
        return error_report(fit_predict(e_va), u_va, sigma)["max_error_over_sigma"]
    except Exception as exc:
        print(f"  {label}: failed ({type(exc).__name__}: {exc}) -> NaN")
        return float("nan")


def gate_word(ratio, gate):
    if not np.isfinite(ratio):
        return "n/a"
    return "PASS" if ratio < gate else "FAIL"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SurrogateParameters.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out, d, g = cfg["output"], cfg["domain"], cfg["gp"]
    sigma = cfg["accuracy_gate"]["sigma_noise_m"]
    gate = cfg["accuracy_gate"]["max_error_over_sigma"]

    # 1. existing solves only; load_responses raises if a CSV is missing
    e_tr, u_tr = load_responses(out["training_csv"])
    e_va, u_va = load_responses(out["validation_csv"])
    if u_tr.shape[1] != u_va.shape[1]:
        raise RuntimeError(f"training has {u_tr.shape[1]} sensor(s), "
                           f"validation has {u_va.shape[1]}")
    names = sensor_labels(out["model_file"], u_tr.shape[1])
    N = e_tr.size
    print(f"training   {out['training_csv']}: {N} solves, "
          f"E {e_tr[0] / 1e9:.3f} .. {e_tr[-1] / 1e9:.3f} GPa")
    print(f"validation {out['validation_csv']}: {e_va.size} solves, "
          f"E {e_va[0] / 1e9:.3f} .. {e_va[-1] / 1e9:.3f} GPa")
    print(f"sensors: {', '.join(names)}")

    # 2a. alpha * u = const  <=>  E * u = const; max relative deviation per sensor
    print("\ncheck a: max |E*u - mean(E*u)| / |mean(E*u)| on the training set")
    eu = e_tr[:, None] * u_tr
    for s, name in enumerate(names):
        m = eu[:, s].mean()
        print(f"  {name:<16}{np.max(np.abs(eu[:, s] - m)) / abs(m):.3e}")

    # 2b. no extrapolation: validation E must sit inside the training E range
    if e_va.min() < e_tr.min() or e_va.max() > e_tr.max():
        raise RuntimeError(
            f"validation E range [{e_va.min():.6e}, {e_va.max():.6e}] Pa is not inside "
            f"the training E range [{e_tr.min():.6e}, {e_tr.max():.6e}] Pa")
    print("check b: validation E range lies inside the training E range")

    rows = []
    for n in SUBSET_SIZES:
        # 3. n equally spaced training points, both ends always included
        idx_f = np.linspace(0, N - 1, n)
        idx = np.rint(idx_f).astype(int)
        if np.any(idx_f != idx):
            raise RuntimeError(f"n = {n}: indices {idx_f} are not exact integers for N = {N}")
        if np.unique(idx).size != n:
            raise RuntimeError(f"n = {n}: indices {idx} are not unique for N = {N}")
        e_s, u_s = e_tr[idx], u_tr[idx]
        print(f"\nn = {n:2d}: training indices {idx.tolist()}")

        # 4. fit the four models on the subset; 6. score them at the validation E
        def gp_fp(e):
            return ResponseSurrogate(d["e_scale_Pa"], d["e_min_Pa"], d["e_max_Pa"],
                                     gp_jitter=g["gp_jitter"],
                                     n_restarts=g["n_restarts_optimizer"],
                                     random_state=g.get("random_state", 20260911)
                                     ).fit(e_s, u_s).predict(e)

        def pchip_fp(e):
            return InterpBaseline(d["e_scale_Pa"]).fit(e_s, u_s).predict(e)

        def hermite_fp(e):
            return HermiteInT(d["e_scale_Pa"]).fit(e_s, u_s).predict(e)

        fitted = {}  # keeps the GP+slope model so its ell_ can be printed

        def gp_slope_fp(e):
            # slopes du/dt = -u, as for the Hermite spline
            fitted["gp_slope"] = GradientEnhancedGP(d["e_scale_Pa"]).fit(e_s, u_s, -u_s)
            return fitted["gp_slope"].predict(e)

        rows.append({"row": "subset", "n": n,
                     "gp": score(gp_fp, e_va, u_va, sigma, "GP"),
                     "pchip": score(pchip_fp, e_va, u_va, sigma, "PCHIP"),
                     "hermite": score(hermite_fp, e_va, u_va, sigma, "Hermite"),
                     "gp_slope": score(gp_slope_fp, e_va, u_va, sigma, "GP+slope")})
        gpm = fitted.get("gp_slope")
        print("  GP+slope ell_: " + (" ".join(f"{v:.3g}" for v in gpm.ell_)
                                     if gpm is not None else "n/a (fit failed)"))

    # 4b. adjoint row: Hermite and GP+slope fitted with the Kratos adjoint slopes
    #     (adjoint_training_solves.py) instead of du/dt = -u
    adjoint_csv = os.path.join(out["dir"], ADJOINT_CSV)
    if not os.path.exists(adjoint_csv):
        print(f"\nadjoint: {adjoint_csv} not found -> no adjoint row")
    else:
        e_a, u_a, dudt_a = load_adjoint(adjoint_csv, u_tr.shape[1])
        # same no-extrapolation rule as check b, against the training E range
        if e_a.min() < e_tr.min() or e_a.max() > e_tr.max():
            raise RuntimeError(
                f"adjoint E range [{e_a.min():.6e}, {e_a.max():.6e}] Pa is not inside "
                f"the training E range [{e_tr.min():.6e}, {e_tr.max():.6e}] Pa")
        print(f"\nadjoint: {adjoint_csv}: {e_a.size} points, "
              f"E {e_a[0] / 1e9:.3f} .. {e_a[-1] / 1e9:.3f} GPa")
        # how far the adjoint slopes are from the formula du/dt = -u
        print("  max |dudt + u| / |u| per sensor")
        dev = np.abs(dudt_a + u_a) / np.abs(u_a)
        for s, name in enumerate(names):
            print(f"  {name:<16}{dev[:, s].max():.3e}")

        def hermite_adj_fp(e):
            return HermiteInT(d["e_scale_Pa"]).fit(e_a, u_a, dudt_train=dudt_a).predict(e)

        fitted = {}  # keeps the GP+slope model so its ell_ can be printed

        def gp_slope_adj_fp(e):
            fitted["gp_slope"] = GradientEnhancedGP(d["e_scale_Pa"]).fit(e_a, u_a, dudt_a)
            return fitted["gp_slope"].predict(e)

        rows.append({"row": "adjoint", "n": e_a.size,
                     "hermite": score(hermite_adj_fp, e_va, u_va, sigma, "Hermite"),
                     "gp_slope": score(gp_slope_adj_fp, e_va, u_va, sigma, "GP+slope")})
        gpm = fitted.get("gp_slope")
        print("  GP+slope ell_: " + (" ".join(f"{v:.3g}" for v in gpm.ell_)
                                     if gpm is not None else "n/a (fit failed)"))

    # 5. analytic reference: one training solve, the one closest to E_ref.
    #    Without Kratos, E_ref is taken from the config's analytic_check_model block.
    e_ref = cfg["analytic_check_model"]["e_ref_Pa"]
    k = int(np.argmin(np.abs(e_tr - e_ref)))
    e0, u0 = e_tr[k], u_tr[k]
    print(f"\nanalytic: single solve at E0 = {e0 / 1e9:.4f} GPa "
          f"(index {k}, E_ref = {e_ref / 1e9:.4f} GPa)")
    rows.append({"row": "analytic", "n": 1,
                 "analytic": score(lambda e: u0[None, :] * (e0 / e)[:, None],
                                   e_va, u_va, sigma, "analytic")})

    # 7. table: max error / sigma and gate verdict per model
    print(f"\nmax error / sigma at {e_va.size} validation points, gate < {gate}")
    head = f"{'row':<10}{'n':>4}" + "".join(f"{lab:>14}{'':>6}" for _, lab in MODELS)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['row']:<10}{r['n']:>4}"
        for key, _ in MODELS:
            if key in r:
                line += f"{r[key]:>14.3e}{gate_word(r[key], gate):>6}"
            else:
                line += f"{'-':>14}{'':>6}"
        print(line)

    os.makedirs(out["dir"], exist_ok=True)
    table_file = os.path.join(out["dir"], "gradient_surrogate_study.csv")
    with open(table_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "n"] + [c for key, _ in MODELS
                                   for c in (f"{key}_max_error_over_sigma", f"{key}_gate")])
        for r in rows:
            cells = [r["row"], r["n"]]
            for key, _ in MODELS:
                cells += ([repr(float(r[key])), gate_word(r[key], gate)]
                          if key in r else ["", ""])
            w.writerow(cells)
    print(f"\ntable -> {table_file}")


if __name__ == "__main__":
    main()
