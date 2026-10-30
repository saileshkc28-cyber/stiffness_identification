r"""Gradient-enhanced surrogate from an existing case folder, into a new one.

Reads only, from --source (a case written by run_surrogate_sweep.py):
    SurrogateParameters.json                 domain, accuracy gate, output paths
    surrogate\gradient_training_responses.csv  E, u, du/dt (adjoint_training_solves.py)
    surrogate\validation_responses.csv       validation solves
    surrogate\response_surrogate.joblib(.json)  sensor names, sanity comparison
    sensor_data.json                         sensor names, if the .json has none
    HierarchicalParameters.json              copied with two paths changed

Writes, into --out only:
    surrogate\response_surrogate.joblib(.json)  GradientResponseSurrogate
    surrogate\validation_report.json         BuildSurrogate.py layout + solve counts
    HierarchicalParameters.json              surrogate_file and output_path -> --out

No solves, no Kratos. Usage:
    python build_gradient_surrogate.py --source <case>\N01 --out <case>\N01_adjoint
    python build_gradient_surrogate.py --source ... --out ... --overwrite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from gradient_response_surrogate import GradientResponseSurrogate
# gradient_surrogate_study imports BuildSurrogate, which imports Kratos only
# inside build_forward_model, so this is safe
from gradient_surrogate_study import ADJOINT_CSV, load_adjoint, load_responses
from response_surrogate import InterpBaseline, ResponseSurrogate, error_report

HERE = Path(__file__).resolve().parent
HIER_DIR = HERE / "hierarchical"          # MainHierarchical's working directory
N_SANITY = 200                            # log-spaced E for the old-vs-new check

# the only two keys changed in HierarchicalParameters.json
CHANGED_KEYS = (("likelihood", "surrogate_file"), ("output", "output_path"))
# paths the hierarchical run only reads; allowed to stay inside --source
READ_ONLY_KEYS = {("likelihood", "observations_file")}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def resolve_in(base, p):
    """Absolute path; a relative p is taken relative to base."""
    p = Path(p)
    return (p if p.is_absolute() else Path(base) / p).resolve()


def inside(p, root):
    """p equals root or lies below it."""
    return Path(p).resolve().is_relative_to(Path(root).resolve())


def guard_write(path, src):
    """Last check before every write: never into --source."""
    if inside(path, src):
        raise RuntimeError(f"refusing to write {path}: it lies inside --source {src}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def sensor_names_from_source(model_file, src):
    """Sensor names and sensor file, from the saved surrogate's .json, else from
    <source>\\sensor_data.json. Both present and different -> error."""
    meta = Path(str(model_file) + ".json")
    sensor_json = src / "sensor_data.json"
    ident = read_json(meta) if meta.exists() else {}
    names_meta = ident.get("sensor_names")
    names_file = ([s["name"] for s in read_json(sensor_json)["list_of_sensors"]]
                  if sensor_json.exists() else None)
    if names_meta and names_file and names_meta != names_file:
        raise RuntimeError(f"sensor names differ: {meta} {names_meta} vs "
                           f"{sensor_json} {names_file}")
    names = names_meta or names_file
    if not names:
        raise RuntimeError(f"no sensor names in {meta} or {sensor_json}")
    sensor_file = ident.get("sensor_data_file") or (str(sensor_json)
                                                    if sensor_json.exists() else None)
    return names, sensor_file, (str(meta) if names_meta else str(sensor_json))


def iter_strings(node, key=()):
    """(key path, value) for every string leaf of a JSON tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from iter_strings(v, key + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_strings(v, key + (i,))
    elif isinstance(node, str):
        yield key, node


def iter_changes(old, new, key=()):
    """(key path, old, new) for every leaf that differs."""
    if isinstance(old, dict) and isinstance(new, dict):
        for k in list(old) + [k for k in new if k not in old]:
            yield from iter_changes(old.get(k), new.get(k), key + (k,))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            yield from iter_changes(a, b, key + (i,))
    elif old != new:
        yield key, old, new


def new_hierarchical_config(hier, src, out, model_name):
    """Copy of the source config with only surrogate_file and output_path moved
    into --out. Stops if any other path inside --source could be written to."""
    for k in CHANGED_KEYS:
        if k[1] not in hier.get(k[0], {}):
            raise RuntimeError(f"source HierarchicalParameters.json has no {'.'.join(k)}")

    # every other path-like string: relative ones resolve against the run's cwd
    blocked, read_only = [], []
    for key, value in iter_strings(hier):
        if key in CHANGED_KEYS:
            continue
        if not (os.path.isabs(value) or "/" in value or "\\" in value):
            continue
        if inside(resolve_in(HIER_DIR, value), src):
            (read_only if key in READ_ONLY_KEYS else blocked).append((key, value))
    if blocked:
        sys.exit("STOP: HierarchicalParameters.json has path(s) inside --source other than "
                 "surrogate_file and output_path, and I cannot tell whether the run writes "
                 "to them:\n" + "\n".join(f"  {'.'.join(map(str, k))} = {v}"
                                          for k, v in blocked)
                 + "\nNothing was written. Decide how these should be mapped.")
    for key, value in read_only:
        print(f"  note: {'.'.join(map(str, key))} stays inside --source (read only): {value}")

    new = json.loads(json.dumps(hier))
    new["likelihood"]["surrogate_file"] = str(out / "surrogate" / model_name)
    # output folder: same place relative to the case folder, else same folder name
    old_out = resolve_in(HIER_DIR, hier["output"]["output_path"])
    rel = old_out.relative_to(src) if inside(old_out, src) else Path(old_out.name)
    new["output"]["output_path"] = str(out / rel)
    return new


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="gradient-enhanced surrogate into a new case")
    ap.add_argument("--source", required=True, type=Path, help="existing case folder")
    ap.add_argument("--out", required=True, type=Path, help="new case folder")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow an existing --out (files in it are replaced, none deleted)")
    args = ap.parse_args()

    # 1. folder rules, before anything is read or written
    src, out = args.source.resolve(), args.out.resolve()
    if not src.is_dir():
        sys.exit(f"--source {src} is not a folder")
    if inside(out, src):
        sys.exit(f"--out {out} equals --source or lies inside it")
    if out.exists() and not args.overwrite:
        sys.exit(f"--out {out} exists; pass --overwrite to write into it")

    # 2. inputs from --source; relative config paths taken relative to the case folder
    surr_cfg = read_json(src / "SurrogateParameters.json")
    d, ocfg = surr_cfg["domain"], surr_cfg["output"]
    sigma = surr_cfg["accuracy_gate"]["sigma_noise_m"]
    gate = surr_cfg["accuracy_gate"]["max_error_over_sigma"]
    surr_dir = resolve_in(src, ocfg["dir"])
    model_file = resolve_in(src, ocfg["model_file"])
    valid_csv = resolve_in(src, ocfg["validation_csv"])
    grad_csv = surr_dir / ADJOINT_CSV
    for p in (surr_dir, model_file, valid_csv):
        if not inside(p, src):
            sys.exit(f"SurrogateParameters.json points outside --source: {p}")

    e_va, u_va = load_responses(valid_csv)
    n_sensors = u_va.shape[1]
    e_a, u_a, dudt_a = load_adjoint(grad_csv, n_sensors)   # raises on any status != ok
    names, sensor_file, names_from = sensor_names_from_source(model_file, src)
    if len(names) != n_sensors:
        sys.exit(f"{len(names)} sensor names but {n_sensors} response column(s)")
    hier = read_json(src / "HierarchicalParameters.json")

    print(f"source      : {src}")
    print(f"out         : {out}")
    print(f"adjoint     : {grad_csv}: {e_a.size} points, "
          f"E {e_a[0] / 1e9:.3f} .. {e_a[-1] / 1e9:.3f} GPa")
    print(f"validation  : {valid_csv}: {e_va.size} points")
    print(f"sensors     : {', '.join(names)}  (from {names_from})")
    print(f"domain      : [{d['e_min_Pa'] / 1e9:g}, {d['e_max_Pa'] / 1e9:g}] GPa, "
          f"e_scale {d['e_scale_Pa'] / 1e9:g} GPa")

    # 7a. new hierarchical config, checked before anything is written
    new_hier = new_hierarchical_config(hier, src, out, model_file.name)

    # 3. fit on the adjoint points only
    model = GradientResponseSurrogate(
        d["e_scale_Pa"], d["e_min_Pa"], d["e_max_Pa"],
        gp_jitter=surr_cfg["gp"]["gp_jitter"]).fit(
        e_a, u_a, dudt_a, sensor_names=names, sensor_data_file=sensor_file,
        training_csv=str(grad_csv))
    print("ell_        : " + " ".join(f"{v:.4g}" for v in model.identity_["ell_"]))
    print("jitter_     : " + " ".join(f"{v:.0e}" for v in model.identity_["jitter_"]))

    # 4. score at the validation E; PCHIP on the same points, as BuildSurrogate does
    base = InterpBaseline(d["e_scale_Pa"]).fit(e_a, u_a)
    rep = {
        "gp": error_report(model.predict(e_va), u_va, sigma),
        "interp_baseline": error_report(base.predict(e_va), u_va, sigma),
        "gate_max_error_over_sigma": gate,
        "n_training": int(e_a.size),
        "n_validation": int(e_va.size),
        "solves_used_this_session": 0,
        # behind this surrogate: one primal per adjoint point (adjoint_training_solves.py),
        # one per validation point (BuildSurrogate.py), one adjoint per point and sensor
        "n_forward_solves": int(e_a.size + e_va.size),
        "n_forward_solves_training": int(e_a.size),
        "n_forward_solves_validation": int(e_va.size),
        "n_adjoint_solves": int(e_a.size * n_sensors),
        "identity": model.identity_,
        "response_model": "kratos",
        "slope_source": "kratos_adjoint",
        "source_case": str(src),
        "training_csv": str(grad_csv),
        "validation_csv": str(valid_csv),
    }
    rep["gp_passes_gate"] = rep["gp"]["max_error_over_sigma"] < gate
    rep["interp_passes_gate"] = rep["interp_baseline"]["max_error_over_sigma"] < gate

    print(f"\n{'model':<18}{'max|err| [m]':>16}{'max err / sigma':>18}{'gate':>8}")
    for label, key in (("GP+slope", "gp"), ("PCHIP baseline", "interp_baseline")):
        r = rep[key]
        ok = "PASS" if r["max_error_over_sigma"] < gate else "FAIL"
        print(f"{label:<18}{r['max_abs_error_m']:>16.3e}"
              f"{r['max_error_over_sigma']:>18.3e}{ok:>8}")
    print(f"gate: max error / sigma < {gate}")

    # 5. sanity: new vs the source surrogate on a log-spaced grid in both domains
    old = ResponseSurrogate.load(str(model_file))
    lo, hi = max(model.e_min, old.e_min), min(model.e_max, old.e_max)
    e_chk = np.geomspace(lo, hi, N_SANITY)
    cmp = error_report(model.predict(e_chk), old.predict(e_chk), sigma)
    rep["comparison_to_source_surrogate"] = {
        "source_model_file": str(model_file),
        "n_points": N_SANITY, "e_min_Pa": lo, "e_max_Pa": hi,
        "max_abs_diff_over_sigma": cmp["max_error_over_sigma"],
        "per_sensor_max_abs_diff_over_sigma": cmp["per_sensor_max_over_sigma"],
    }
    print(f"\nsanity: max |new - old| / sigma over {N_SANITY} log-spaced E in "
          f"[{lo / 1e9:g}, {hi / 1e9:g}] GPa = {cmp['max_error_over_sigma']:.3e}")
    if resolve_in(HIER_DIR, hier["likelihood"]["surrogate_file"]) != model_file:
        print(f"  note: source HierarchicalParameters.json surrogate_file is "
              f"{hier['likelihood']['surrogate_file']}, compared against {model_file}")

    # 6. model and report into --out\surrogate
    out_surr = out / "surrogate"
    new_model = out_surr / model_file.name
    report_file = out_surr / "validation_report.json"
    for p in (new_model, Path(str(new_model) + ".json"), report_file):
        guard_write(p, src)
    os.makedirs(out_surr, exist_ok=True)
    model.save(str(new_model))
    with open(report_file, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nmodel  -> {new_model}\nreport -> {report_file}")

    # 8. gate failed: report is on file, no hierarchical config
    if not rep["gp_passes_gate"]:
        sys.exit("GP+slope FAILED the accuracy gate -- HierarchicalParameters.json not "
                 "written, do not run the hierarchical stage on this surrogate")

    # 7. hierarchical config, only the two keys changed
    hier_file = out / "HierarchicalParameters.json"
    guard_write(hier_file, src)
    changes = list(iter_changes(hier, new_hier))
    if {k for k, _, _ in changes} - set(CHANGED_KEYS):
        raise RuntimeError(f"unexpected change(s) in the hierarchical config: {changes}")
    with open(hier_file, "w") as f:
        json.dump(new_hier, f, indent=4)
    print(f"\n{hier_file}: changed key(s)")
    for key, a, b in changes:
        print(f"  {'.'.join(map(str, key))}\n    old {a}\n    new {b}")

    # 9. how to start the run (PowerShell; MainHierarchical needs Kratos)
    print("\nstart the hierarchical run with:")
    print(f'  Set-Location "{HIER_DIR}"')
    print(f'  & "{sys.executable}" MainHierarchical.py "{hier_file}"')


if __name__ == "__main__":
    main()
