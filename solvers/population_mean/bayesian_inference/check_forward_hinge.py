"""Check the Phase 2 forward model for spurious solves across the prior range.

Diagnostic only. Builds KratosForwardModel once, exactly as BayesianAnalysis does
(forward_model and parameters blocks of BayesianParameters.json), with the primal
parameter file swapped for the one given, and evaluates

    200 alpha values evenly spaced over the prior [0.2, 2.0]
  + the 17 alpha values of the bad Phase 1 specimens

For a linear static model alpha * u is constant, so any alpha whose alpha * u deviates
from the median by more than REL_TOL is a spurious solve.

    python check_forward_hinge.py --primal PrimalParametersBayes.json --label fixed
    python check_forward_hinge.py --compare diagnostics/forward_hinge_original.csv \
                                            diagnostics/forward_hinge_fixed.csv
"""
import argparse
import csv
import os
import time

import numpy as np

PHASE1_CSV = "../damaged_system/phase1_distribution_runs/phase1_samples.csv"
BAD_IDS = [59, 162, 212, 221, 271, 396, 444, 585, 586, 622, 634, 654, 659, 690,
           772, 843, 982]
N_GRID = 200
REL_TOL = 1e-6
OUT_DIR = "diagnostics"


def alpha_values():
    grid = [("grid", None, float(a)) for a in np.linspace(0.2, 2.0, N_GRID)]
    with open(PHASE1_CSV, newline="") as f:
        by_id = {int(r["sample_id"]): float(r["alpha_true"]) for r in csv.DictReader(f)}
    return grid + [("bad_specimen", sid, by_id[sid]) for sid in BAD_IDS]


def run(primal_file, label):
    import KratosMultiphysics as Kratos
    from kratos_forward_model import KratosForwardModel

    with open("BayesianParameters.json") as f:
        project = Kratos.Parameters(f.read())
    settings = project["forward_model"].Clone()
    settings["primal_parameters_file"].SetString(primal_file)
    entries = [project["parameters"][i] for i in range(project["parameters"].size())]

    model = Kratos.Model()
    forward = KratosForwardModel(model, settings, entries)

    records = []
    for kind, sid, alpha in alpha_values():
        start = time.perf_counter()
        try:
            u = np.atleast_1d(np.asarray(forward.Evaluate([alpha]), dtype=float))
            status = "ok"
        except Exception as exc:
            u = np.full(len(forward.located), np.nan)
            status = " ".join(f"failed: {type(exc).__name__}: {exc}".split())[:300]
        records.append({"kind": kind, "sample_id": "" if sid is None else sid,
                        "alpha": alpha, "status": status,
                        "seconds": time.perf_counter() - start,
                        **{f"u_{j}": float(v) for j, v in enumerate(u)}})
    forward.Finalize()

    scaled = np.array([[r["alpha"] * r[f"u_{j}"] for j in range(len(forward.located))]
                       for r in records])
    constant = np.nanmedian(scaled, axis=0)
    for r, s in zip(records, scaled):
        dev = np.abs(s / constant - 1.0)
        r["max_rel_dev"] = float(np.nanmax(dev)) if r["status"] == "ok" else float("nan")
        r["spurious"] = bool(r["status"] != "ok" or np.any(dev > REL_TOL))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"forward_hinge_{label}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    bad = [r for r in records if r["spurious"]]
    print(f"\nprimal file      : {primal_file}")
    print(f"evaluations      : {len(records)}   forward solves: {forward.n_solves}")
    print(f"alpha*u constant : {constant}")
    print(f"spurious (> {REL_TOL:g}) : {len(bad)} of {len(records)}  "
          f"(grid {sum(r['kind'] == 'grid' for r in bad)}, "
          f"bad specimens {sum(r['kind'] == 'bad_specimen' for r in bad)}, "
          f"failed {sum(r['status'] != 'ok' for r in bad)})")
    for r in bad:
        print(f"  {r['kind']:<12} {str(r['sample_id']):>4} alpha {r['alpha']:.6f}  "
              f"rel dev {r['max_rel_dev']:.3e}  {r['status'][:80]}")
    ok = [r for r in records if not r["spurious"]]
    if ok:
        print(f"worst rel dev among the rest: {max(r['max_rel_dev'] for r in ok):.2e}")
    print(f"wrote {out}")


def compare(original_csv, fixed_csv):
    def load(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    orig, fixed = load(original_csv), load(fixed_csv)
    if [r["alpha"] for r in orig] != [r["alpha"] for r in fixed]:
        raise SystemExit("the two files do not evaluate the same alpha values")
    cols = [c for c in orig[0] if c.startswith("u_")]
    good = [(o, f) for o, f in zip(orig, fixed)
            if o["spurious"] == "False" and f["spurious"] == "False"]
    diffs = [max(abs(float(o[c]) / float(f[c]) - 1.0) for c in cols) for o, f in good]
    print(f"original spurious : {sum(r['spurious'] == 'True' for r in orig)} of {len(orig)}")
    print(f"fixed spurious    : {sum(r['spurious'] == 'True' for r in fixed)} of {len(fixed)}")
    print(f"values good in both: {len(good)}; largest |u_orig / u_fixed - 1| = "
          f"{max(diffs):.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--primal", default="PrimalParametersBayes.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", nargs=2, metavar=("ORIGINAL_CSV", "FIXED_CSV"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.primal, args.label)


if __name__ == "__main__":
    main()
