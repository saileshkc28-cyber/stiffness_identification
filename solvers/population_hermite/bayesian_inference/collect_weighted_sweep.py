r"""Collects the weighted three-point sensor sweep and draws two figures. Read-only.

    python collect_weighted_sweep.py                    # defaults to Analysis\sensors_weighted
    python collect_weighted_sweep.py D:\...\Analysis\sensors_weighted

Each run inverts three generated measurements, alpha_j = 1 -+ sqrt3 * 0.1 and 1, on the
first k sensors under one instrument sigma (option A), and combines them with weights
1/6, 2/3, 1/6. With one zone and a linear model, u_i = u_i_ref / alpha, so each point's
posterior SD should fall as 1 / sqrt(N_eff) with

    N_eff(k) = sum_{i<=k} u_i^2 / u_tip^2

read from each run's run_info.json (clean u_true of the central point; the 1 / alpha
scaling cancels, so N_eff is the same for all three points). Two levels are reported:

    posterior    each point against its own alpha_true
    population   recovered mean and between-case SD against the prescribed N(1.0, 0.1);
                 the mixture SD adds within-posterior uncertainty and is shown for
                 reference only, it is not the population SD

Every alpha is also given as E = alpha * E_ref [GPa], E_ref from run_info.json
(206.9 GPa).

Writes results.csv, weighted_information.png and weighted_population.png.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else r"D:\KratosProjects\MCMC\Analysis\sensors_weighted")

SIGMA = 3.788e-08
E_REF_GPA = 206.9      # fallback when run_info.json has no e_ref

EDGE = "#1f3b57"
TRUTH = "#d62728"
MIX = "#2ca02c"
NAIVE = "#d85a30"
POINTS = {"point_1_low": "#2a78d6", "point_2_central": EDGE, "point_3_high": "#9467bd"}

FIELDS = ["run", "label", "n_sensors", "stations", "n_eff", "sigma_assumed", "e_ref_GPa",
          "case", "z_value", "weight", "alpha_true", "E_true_GPa",
          "alpha_mean", "alpha_std", "alpha_p2.5", "alpha_p97.5",
          "E_mean_GPa", "E_std_GPa", "E_p2.5_GPa", "E_p97.5_GPa",
          "bias_pct", "z", "covers", "sd_ratio", "sd_pred_ratio", "n_forward_solves",
          "alpha_target_mean", "alpha_target_sd",
          "alpha_recovered_mean", "alpha_between_case_sd", "alpha_total_mixture_sd",
          "E_recovered_mean_GPa", "E_between_case_sd_GPa", "E_total_mixture_sd_GPa",
          "err_mean_pct", "err_between_sd_pct", "err_mixture_sd_pct",
          "total_forward_solves", "wall_time_s"]

SKIPPED = []


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect(folder):
    info_file = folder / "run_info.json"
    combined_file = folder / "combined" / "combined_summary.json"
    cases_file = folder / "combined" / "case_summary.csv"
    if not (info_file.exists() and combined_file.exists() and cases_file.exists()):
        why = "no run_info.json -- the sweep never reached this case"
        if info_file.exists():
            code = json.loads(info_file.read_text()).get("returncode")
            why = ("prepared only, Phase 2 not run" if code is None
                   else f"Phase 2 exited with {code}" if code
                   else "Phase 2 reported success but wrote no combined summary")
            if code is not None and cases_file.exists():
                failed = [r["case_name"] for r in read_csv(cases_file) if r["status"] != "ok"]
                if failed:
                    why = f"{len(failed)} of 3 points failed ({', '.join(failed)}), no combination"
        SKIPPED.append((folder.name, why))
        return None

    info = json.loads(info_file.read_text())
    combined = json.loads(combined_file.read_text())
    e_ref = info.get("e_ref", E_REF_GPA * 1e9) / 1e9
    target_m, target_s = info["alpha_target_mean"], info["alpha_target_sd"]

    points = []
    for r in read_csv(cases_file):
        if r["status"] != "ok":
            continue
        a = float(r["alpha_true"])
        m, s = float(r["alpha_posterior_mean"]), float(r["alpha_posterior_sd"])
        lo, hi = float(r["alpha_ci_2_5"]), float(r["alpha_ci_97_5"])
        points.append({
            "case": r["case_name"],
            "z_value": float(r["z_value"]),
            "weight": float(r["weight"]),
            "alpha_true": a,
            "E_true_GPa": a * e_ref,
            "alpha_mean": m,
            "alpha_std": s,
            "alpha_p2.5": lo,
            "alpha_p97.5": hi,
            "E_mean_GPa": m * e_ref,
            "E_std_GPa": s * e_ref,
            "E_p2.5_GPa": lo * e_ref,
            "E_p97.5_GPa": hi * e_ref,
            "bias_pct": 100.0 * (m - a) / a,
            "z": (m - a) / s,
            "covers": lo <= a <= hi,
            "n_forward_solves": int(r["number_of_forward_solves"]),
        })

    mu = combined["alpha_recovered_mean"]
    between = combined["alpha_between_case_sd"]
    mixture = combined["alpha_total_mixture_sd"]
    return {
        "run": folder.name,
        "label": info["label"],
        "n_sensors": info["n_sensors"],
        "stations": " ".join(format(x, ".1f") for x in info["stations"]),
        "n_eff": info.get("n_eff"),
        "sigma_assumed": info["sigma_assumed"],
        "e_ref_GPa": e_ref,
        "alpha_target_mean": target_m,
        "alpha_target_sd": target_s,
        "alpha_recovered_mean": mu,
        "alpha_between_case_sd": between,
        "alpha_total_mixture_sd": mixture,
        "E_recovered_mean_GPa": mu * e_ref,
        "E_between_case_sd_GPa": between * e_ref,
        "E_total_mixture_sd_GPa": mixture * e_ref,
        "err_mean_pct": 100.0 * (mu - target_m) / target_m,
        "err_between_sd_pct": 100.0 * (between - target_s) / target_s,
        "err_mixture_sd_pct": 100.0 * (mixture - target_s) / target_s,
        "total_forward_solves": combined["total_forward_solves"],
        "wall_time_s": info.get("wall_time_s"),
        "points": points,
    }


def save(fig, out):
    with open(str(out), "wb") as f:
        fig.savefig(f, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_information(runs, out):
    k = np.array([r["n_sensors"] for r in runs], float)
    have_neff = all(r["n_eff"] is not None for r in runs)
    n_eff = np.array([r["n_eff"] if r["n_eff"] is not None else np.nan for r in runs])
    e_ref = runs[0]["e_ref_GPa"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4))
    fig.subplots_adjust(hspace=0.36)

    ax1.plot(k, k, color=NAIVE, ls="--", lw=1.4, label=r"naive  $N_{eff} = k$")
    ax1.plot(k, n_eff, "o-", color=EDGE, ms=7, lw=1.6,
             label=r"$N_{eff}$ from clean $u_{true}$ (central point)")
    for ki, ne in zip(k, n_eff):
        if np.isfinite(ne):
            ax1.annotate(f"{ne:.2f}", (ki, ne), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8, color=EDGE)
    ax1.set_xticks(k)
    ax1.set_xlabel("sensors  k   (tip first, then inward)")
    ax1.set_ylabel(r"effective sensor count  $N_{eff}$")
    ax1.set_title(r"Weighted: $N_{eff} = \sum u_i^2 / u_{tip}^2$ under one absolute sigma",
                  fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, framealpha=0.9)

    worst = []
    for case, color in POINTS.items():
        sd = np.array([next((p["alpha_std"] for p in r["points"] if p["case"] == case), np.nan)
                       for r in runs])
        if not np.isfinite(sd).any():
            continue
        ax2.plot(k, sd, "o-", color=color, ms=7, lw=1.8, label=f"{case}  posterior SD")
        if have_neff and np.isfinite(sd[0]):
            pred = sd[0] / np.sqrt(n_eff / n_eff[0])
            ax2.plot(k, pred, "--", color=color, lw=1.3, alpha=0.8)
            worst.append(np.nanmax(np.abs(100.0 * (sd - pred) / pred)))
    sec = ax2.secondary_yaxis("right", functions=(lambda a: a * e_ref, lambda e: e / e_ref))
    sec.set_ylabel(r"posterior SD of $E$  [GPa]")
    ax2.set_xticks(k)
    ax2.set_xlabel("sensors  k")
    ax2.set_ylabel(r"posterior SD of $\alpha$")
    ax2.set_title(r"Sampled SD per point against the Fisher prediction  $SD_1/\sqrt{N_{eff}}$"
                  " (dashed)", fontsize=12)
    ax2.grid(alpha=0.25)
    handles, _ = ax2.get_legend_handles_labels()
    handles.append(Line2D([], [], color="grey", ls="--", lw=1.3,
                          label=r"predicted  $SD_1/\sqrt{N_{eff}}$  (same colour)"))
    ax2.legend(handles=handles, fontsize=9, framealpha=0.9)
    for ax in (ax1, ax2):
        ax.margins(x=0.06, y=0.15)

    note = (f"max deviation from the Fisher prediction: {max(worst):.1f}%" if worst
            else "no Fisher prediction: N_eff or the first run's SD is missing")
    fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="grey", style="italic")
    save(fig, out)


def figure_population(runs, out):
    k = np.array([r["n_sensors"] for r in runs], float)
    mean = np.array([r["alpha_recovered_mean"] for r in runs])
    between = np.array([r["alpha_between_case_sd"] for r in runs])
    mixture = np.array([r["alpha_total_mixture_sd"] for r in runs])
    target_m, target_s = runs[0]["alpha_target_mean"], runs[0]["alpha_target_sd"]
    e_ref = runs[0]["e_ref_GPa"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4))
    fig.subplots_adjust(hspace=0.36)

    ax1.axhline(target_m, color=TRUTH, ls="--", lw=1.6)
    ax1.plot(k, mean, "o-", color=EDGE, ms=7, lw=1.8)
    for ki, v, r in zip(k, mean, runs):
        ax1.annotate(f"{v:.4f}  ({r['err_mean_pct']:+.2f}%)", (ki, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=EDGE)
    sec = ax1.secondary_yaxis("right", functions=(lambda a: a * e_ref, lambda e: e / e_ref))
    sec.set_ylabel(r"$E$  [GPa]")
    ax1.set_xticks(k)
    ax1.set_xlabel("sensors  k")
    ax1.set_ylabel(r"recovered mean  $\sum w_j m_j$")
    ax1.set_title("Recovered population mean", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(handles=[
        Line2D([], [], color=TRUTH, ls="--", lw=1.6,
               label=f"target  {target_m:.4f}  ({target_m * e_ref:.2f} GPa)"),
        Line2D([], [], marker="o", color=EDGE, ms=7, lw=1.8, label="recovered mean"),
    ], fontsize=9, framealpha=0.9)

    ax2.axhline(target_s, color=TRUTH, ls="--", lw=1.6)
    ax2.plot(k, between, "o-", color=EDGE, ms=7, lw=1.8)
    ax2.plot(k, mixture, "s--", color=MIX, ms=6, lw=1.4)
    for ki, v, r in zip(k, between, runs):
        ax2.annotate(f"{v:.4f}  ({r['err_between_sd_pct']:+.2f}%)", (ki, v),
                     textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8,
                     color=EDGE)
    sec = ax2.secondary_yaxis("right", functions=(lambda a: a * e_ref, lambda e: e / e_ref))
    sec.set_ylabel(r"SD of $E$  [GPa]")
    ax2.set_xticks(k)
    ax2.set_xlabel("sensors  k")
    ax2.set_ylabel(r"SD of $\alpha$")
    ax2.set_title("Recovered population SD (between-case) and total mixture SD", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.legend(handles=[
        Line2D([], [], color=TRUTH, ls="--", lw=1.6,
               label=f"target  {target_s:.4f}  ({target_s * e_ref:.2f} GPa)"),
        Line2D([], [], marker="o", color=EDGE, ms=7, lw=1.8,
               label="between-case SD  (population)"),
        Line2D([], [], marker="s", color=MIX, ms=6, lw=1.4, ls="--",
               label="total mixture SD  (population + inference, not the population SD)"),
    ], fontsize=9, framealpha=0.9)
    for ax in (ax1, ax2):
        ax.margins(x=0.08, y=0.15)
    save(fig, out)


def main():
    if not ARCHIVE.exists():
        sys.exit(f"no such archive: {ARCHIVE}")

    folders = [f for f in sorted(ARCHIVE.iterdir()) if f.is_dir()]
    runs = [r for r in (collect(f) for f in folders) if r]
    if SKIPPED:
        print(f"*** {len(SKIPPED)} of {len(folders)} run folders have no result "
              f"and are NOT in the tables below ***")
        for name, why in SKIPPED:
            print(f"    {name:<20} {why}")
        print()
    if not runs:
        sys.exit(f"no combined/combined_summary.json found under {ARCHIVE}")
    runs.sort(key=lambda r: r["n_sensors"])

    sig = {r["sigma_assumed"] for r in runs}
    if len(sig) > 1:
        print("WARNING: sigma_assumed varies across runs -- this sweep should hold it fixed")
    elif abs(sig.pop() - SIGMA) / SIGMA > 1e-3:
        print(f"WARNING: sigma_assumed is not {SIGMA:.4e}")
    if len({(r["alpha_target_mean"], r["alpha_target_sd"]) for r in runs}) > 1:
        print("WARNING: target values differ between runs -- mixed configs?")
    if any(r["n_eff"] is None for r in runs):
        print("WARNING: N_eff missing in some run_info.json -- no Fisher prediction for those")

    base = runs[0]
    if base["n_sensors"] != 1:
        print("WARNING: no N=1 run -- SD ratios are relative to the smallest run present")
    base_sd = {p["case"]: p["alpha_std"] for p in base["points"]}
    for r in runs:
        pred = (float(np.sqrt(base["n_eff"] / r["n_eff"]))
                if base["n_eff"] is not None and r["n_eff"] is not None else None)
        for p in r["points"]:
            p["sd_ratio"] = p["alpha_std"] / base_sd[p["case"]] if p["case"] in base_sd else None
            p["sd_pred_ratio"] = pred

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in runs:
            for p in r["points"]:
                row = {**r, **p}
                writer.writerow({k: row[k] for k in FIELDS})

    target_m, target_s = base["alpha_target_mean"], base["alpha_target_sd"]
    e_ref = base["e_ref_GPa"]
    print(f"E_ref = {e_ref:g} GPa")
    print(f"target: alpha ~ N({target_m:.4f}, {target_s:.4f}) = "
          f"N({target_m * e_ref:.2f}, {target_s * e_ref:.2f}) GPa\n")

    def num(v, spec):
        return "" if v is None else format(v, spec)

    header = (f"{'run':<6}{'k':>4}{'N_eff':>8}{'point':>17}{'true':>9}{'alpha':>9}{'sd':>9}"
              f"{'95% CI alpha':>20}{'E [GPa]':>10}{'sd':>8}{'bias%':>8}{'z':>7}{'cov':>5}"
              f"{'SD/SD1':>8}{'pred':>7}")
    print("posterior")
    print(header)
    print("-" * len(header))
    for r in runs:
        for p in r["points"]:
            ci_a = f"[{p['alpha_p2.5']:.4f},{p['alpha_p97.5']:.4f}]"
            print(f"{r['label']:<6}{r['n_sensors']:>4}{num(r['n_eff'], '>8.2f'):>8}"
                  f"{p['case']:>17}{p['alpha_true']:>9.4f}{p['alpha_mean']:>9.4f}"
                  f"{p['alpha_std']:>9.4f}{ci_a:>20}{p['E_mean_GPa']:>10.2f}"
                  f"{p['E_std_GPa']:>8.2f}{p['bias_pct']:>8.2f}{p['z']:>7.2f}"
                  f"{'y' if p['covers'] else 'N':>5}{num(p['sd_ratio'], '>8.3f'):>8}"
                  f"{num(p['sd_pred_ratio'], '>7.3f'):>7}")

    header = (f"{'run':<6}{'k':>4}{'mean':>9}{'err%':>8}{'between sd':>12}{'err%':>8}"
              f"{'mixture sd':>12}{'err%':>8}{'E [GPa]':>10}{'betw GPa':>10}{'mix GPa':>9}"
              f"{'solves':>8}{'time_s':>8}")
    print(f"\npopulation   (target mean {target_m:.4f}, sd {target_s:.4f}; "
          f"between-case sd is the population sd)")
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['label']:<6}{r['n_sensors']:>4}{r['alpha_recovered_mean']:>9.4f}"
              f"{r['err_mean_pct']:>8.2f}{r['alpha_between_case_sd']:>12.4f}"
              f"{r['err_between_sd_pct']:>8.2f}{r['alpha_total_mixture_sd']:>12.4f}"
              f"{r['err_mixture_sd_pct']:>8.2f}{r['E_recovered_mean_GPa']:>10.2f}"
              f"{r['E_between_case_sd_GPa']:>10.2f}{r['E_total_mixture_sd_GPa']:>9.2f}"
              f"{r['total_forward_solves']:>8}{num(r['wall_time_s'], '>8.0f'):>8}")

    last = runs[-1]
    if last["n_eff"] is not None:
        print(f"\nN_eff reaches {last['n_eff']:.2f} with {last['n_sensors']} sensors "
              f"(naive would be {last['n_sensors']})")
    for p in last["points"]:
        if p["sd_ratio"] is not None and p["sd_pred_ratio"] is not None:
            print(f"{p['case']:<16} SD shrinks {1 / p['sd_ratio']:.2f}x, Fisher predicts "
                  f"{1 / p['sd_pred_ratio']:.2f}x, naive 1/sqrt(k) predicts "
                  f"{np.sqrt(last['n_sensors'] / base['n_sensors']):.2f}x")
    n_points = sum(len(r["points"]) for r in runs)
    print(f"coverage: {sum(p['covers'] for r in runs for p in r['points'])}/{n_points} "
          f"point 95% intervals contain their alpha_true")

    figure_information(runs, ARCHIVE / "weighted_information.png")
    figure_population(runs, ARCHIVE / "weighted_population.png")
    print(f"\nwrote {out}\n      {ARCHIVE / 'weighted_information.png'}"
          f"\n      {ARCHIVE / 'weighted_population.png'}")


if __name__ == "__main__":
    main()
