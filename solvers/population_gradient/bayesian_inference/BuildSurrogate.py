"""Offline construction of the response surrogate u(E).

Usage:
    python BuildSurrogate.py                  full run through Kratos
    python BuildSurrogate.py --analytic       verification model, no Kratos
    python BuildSurrogate.py --check          repeatability check only
    python BuildSurrogate.py --stage solves   stop after the exact solves
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from response_surrogate import InterpBaseline, ResponseSurrogate, error_report


# --------------------------------------------------------------------------
# Exact forward evaluation
# --------------------------------------------------------------------------
class ExactResponse:
    """E in Pa -> clean sensor vector, via the existing alpha interface."""

    def __init__(self, forward_model, e_ref_Pa):
        self.fm = forward_model
        self.e_ref = float(e_ref_Pa)
        self.attempted = 0
        self.completed = 0

    def __call__(self, e_Pa):
        e = float(e_Pa)
        if not np.isfinite(e) or e <= 0.0:
            raise ValueError(f"E must be finite and positive, got {e}")
        alpha = e / self.e_ref
        self.attempted += 1
        u = np.asarray(self.fm.Evaluate([alpha]), dtype=float).ravel()
        if not np.all(np.isfinite(u)):
            raise RuntimeError(f"non-finite response at E = {e:.6e} Pa")
        self.completed += 1
        return u


class AnalyticResponse:
    """u = u_ref * E_ref / E. Cheap verification model only -- never production."""

    def __init__(self, u_ref_m, e_ref_Pa):
        self.u_ref = np.asarray(u_ref_m, dtype=float).ravel()
        self.e_ref = float(e_ref_Pa)
        self.attempted = 0
        self.completed = 0

    def __call__(self, e_Pa):
        self.attempted += 1
        u = self.u_ref * (self.e_ref / float(e_Pa))
        self.completed += 1
        return u


def build_forward_model(base_config_path):
    """Construct KratosForwardModel exactly as BayesianAnalysis.Initialize does.
    Only the forward_model and parameters blocks are read; likelihood is ignored."""
    import KratosMultiphysics as Kratos
    from kratos_forward_model import KratosForwardModel

    with open(base_config_path) as f:
        settings = Kratos.Parameters(f.read())

    for block in ("forward_model", "parameters"):
        if not settings.Has(block):
            raise RuntimeError(f"'{block}' missing from {base_config_path}")

    entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
    if len(entries) != 1:
        raise RuntimeError(
            f"surrogate is one-dimensional in E, but 'parameters' has {len(entries)} "
            "entries -- the multi-zone surrogate is a separate design"
        )

    model = Kratos.Model()
    fm = KratosForwardModel(model, settings["forward_model"], entries)
    if fm.refs.size != 1:
        raise RuntimeError(f"expected one reference value, got {fm.refs.size}")

    # KratosForwardModel has filled in the default, so the key is always there
    sensor_file = settings["forward_model"]["sensor_data_file"].GetString()
    with open(sensor_file) as f:
        names = [s["name"] for s in json.load(f)["list_of_sensors"]]
    if len(names) != len(fm.located):
        raise RuntimeError(f"{sensor_file} lists {len(names)} sensors, "
                           f"the forward model located {len(fm.located)}")
    return fm, float(fm.refs[0]), sensor_file, names


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def design_points(cfg):
    d = cfg["domain"]
    e_scale, e_min, e_max = d["e_scale_Pa"], d["e_min_Pa"], d["e_max_Pa"]
    a, b = np.log(e_min / e_scale), np.log(e_max / e_scale)

    n_tr = cfg["design"]["n_training"]
    n_va = cfg["design"]["n_validation"]
    t_train = np.linspace(a, b, n_tr)

    rng = np.random.default_rng(cfg["design"]["design_seed"])
    span = b - a
    edges = np.array([a + 0.02 * span, b - 0.02 * span])
    interior = rng.uniform(a + 0.02 * span, b - 0.02 * span, size=n_va - edges.size)
    t_valid = np.sort(np.concatenate([edges, interior]))

    return e_scale * np.exp(t_train), e_scale * np.exp(t_valid)


# --------------------------------------------------------------------------
# Resumable solve driver
# --------------------------------------------------------------------------
def to_float(x):
    """numpy 2 reprs as 'np.float64(1.0)'; accept those as well as plain text."""
    if isinstance(x, str) and x.startswith("np.float"):
        x = x[x.index("(") + 1:x.rindex(")")]
    return float(x)


def load_done(path, n_sensors):
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # columns are u_0, u_1, ... in sensor-file order; names are not in the file
        cols = [c for c in (reader.fieldnames or []) if c.startswith("u_")]
        if cols and len(cols) != n_sensors:
            raise RuntimeError(
                f"{path} holds {len(cols)} response column(s) but the current sensor "
                f"file has {n_sensors} sensor(s). Use a fresh output folder; the "
                "existing file is left as it is.")
        for row in reader:
            if row.get("status") != "ok":
                continue
            u = [to_float(v) for k, v in row.items() if k.startswith("u_")]
            done[to_float(row["E_Pa"])] = np.array(u)
    return done


def run_points(responder, e_values, csv_path, budget, label, n_sensors):
    done = load_done(csv_path, n_sensors)
    todo = [e for e in e_values if not any(abs(e - k) <= 1e-6 * e for k in done)]
    print(f"[{label}] {len(done)} already on file, {len(todo)} to solve")

    new_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = None
        for e in todo:
            if budget["used"] >= budget["max"]:
                raise SystemExit(
                    f"solve budget exhausted at {budget['used']}/{budget['max']} "
                    f"-- raise max_new_solves deliberately, do not loosen tolerances"
                )
            budget["used"] += 1
            u = responder(e)
            if writer is None:
                cols = ["E_Pa"] + [f"u_{i}" for i in range(u.size)] + ["status"]
                writer = csv.writer(f)
                if new_header:
                    writer.writerow(cols)
            writer.writerow([repr(float(e))]
                            + [repr(float(v)) for v in u] + ["ok"])
            f.flush()
            done[e] = u
            print(f"  E = {e / 1e9:8.3f} GPa  ->  u = {np.array2string(u, precision=6)}")

    keys = sorted(done)
    return np.array(keys), np.vstack([done[k] for k in keys])


def repeatability_check(responder, e_values):
    first = np.vstack([responder(e) for e in e_values])
    second = np.vstack([responder(e) for e in reversed(e_values)])[::-1]
    d = np.max(np.abs(first - second) / np.maximum(np.abs(first), 1e-300))
    print(f"repeatability: max relative difference {d:.3e} over {len(e_values)} values")
    return d


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SurrogateParameters.json")
    ap.add_argument("--analytic", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", choices=["solves", "fit", "all"], default="all")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out = dict(cfg["output"])
    if args.analytic:
        stem = out["dir"]
        out = {k: (v.replace(stem, stem + "_analytic", 1) if isinstance(v, str) else v)
               for k, v in out.items()}
    os.makedirs(out["dir"], exist_ok=True)

    if args.analytic:
        am = cfg["analytic_check_model"]
        responder = AnalyticResponse(am["u_ref_m"], am["e_ref_Pa"])
        e_ref = am["e_ref_Pa"]
        sensor_file, sensor_names = None, None
        n_sensors = responder.u_ref.size
        print("VERIFICATION MODEL in use (u = u_ref * E_ref / E) -- not production")
    else:
        fm, e_ref, sensor_file, sensor_names = build_forward_model(cfg["base_config"])
        responder = ExactResponse(fm, e_ref)
        n_sensors = len(sensor_names)
        print(f"Kratos forward model built, E_ref read from model = {e_ref:.6e} Pa "
              f"({e_ref / 1e9:.4f} GPa), {len(fm.located)} sensor(s)")
        print(f"sensor file {sensor_file}: {', '.join(sensor_names)}")

    e_train, e_valid = design_points(cfg)
    budget = {"used": 0, "max": cfg["budget"]["max_new_solves"]}

    try:
        if args.check:
            probe = [float(e_train[0]), float(e_train[len(e_train) // 2]),
                     float(e_train[-1])]
            repeatability_check(responder, probe)
            return
        e_tr, u_tr = run_points(responder, e_train, out["training_csv"], budget, "training",
                                n_sensors)
        e_va, u_va = run_points(responder, e_valid, out["validation_csv"], budget,
                                "validation", n_sensors)
    finally:
        if not args.analytic:
            responder.fm.Finalize()
    print(f"solves used this session: {budget['used']} / {budget['max']} "
          f"(attempted {responder.attempted}, completed {responder.completed})")
    if args.stage == "solves":
        return

    d, g = cfg["domain"], cfg["gp"]
    gp = ResponseSurrogate(d["e_scale_Pa"], d["e_min_Pa"], d["e_max_Pa"],
                           gp_jitter=g["gp_jitter"],
                           n_restarts=g["n_restarts_optimizer"],
                           sensor_names=sensor_names, sensor_data_file=sensor_file,
                           random_state=g.get("random_state", 20260911)).fit(e_tr, u_tr)
    base = InterpBaseline(d["e_scale_Pa"]).fit(e_tr, u_tr)
    if args.stage == "fit":
        gp.save(out["model_file"])
        return

    sigma = cfg["accuracy_gate"]["sigma_noise_m"]
    gate = cfg["accuracy_gate"]["max_error_over_sigma"]
    rep = {
        "gp": error_report(gp.predict(e_va), u_va, sigma),
        "interp_baseline": error_report(base.predict(e_va), u_va, sigma),
        "gate_max_error_over_sigma": gate,
        "n_training": int(e_tr.size),
        "n_validation": int(e_va.size),
        "solves_used_this_session": budget["used"],
        "identity": gp.identity_,
        "response_model": "analytic" if args.analytic else "kratos",
    }
    rep["gp_passes_gate"] = rep["gp"]["max_error_over_sigma"] < gate
    rep["interp_passes_gate"] = rep["interp_baseline"]["max_error_over_sigma"] < gate

    gp.save(out["model_file"])
    with open(out["report_file"], "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n{'model':<18}{'max|err| [m]':>16}{'max err / sigma':>18}{'gate':>8}")
    for name, key in (("GP Matern 5/2", "gp"), ("PCHIP baseline", "interp_baseline")):
        r = rep[key]
        ok = "PASS" if r["max_error_over_sigma"] < gate else "FAIL"
        print(f"{name:<18}{r['max_abs_error_m']:>16.3e}"
              f"{r['max_error_over_sigma']:>18.3e}{ok:>8}")
    print(f"\nreport -> {out['report_file']}\nmodel  -> {out['model_file']}")
    if not rep["gp_passes_gate"]:
        sys.exit("GP failed the accuracy gate -- add training points, do not proceed")


if __name__ == "__main__":
    main()
