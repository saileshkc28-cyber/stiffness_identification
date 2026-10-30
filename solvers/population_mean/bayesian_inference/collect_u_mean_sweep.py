r"""Collects the mean-only (u_mean) sensor sweep and draws two figures. Read-only.

    python collect_u_mean_sweep.py                    # defaults to Analysis\sensors_u_mean
    python collect_u_mean_sweep.py D:\...\Analysis\sensors_u_mean

Each run inverts the population-mean response of the first k sensors under one
instrument sigma (option A). With one zone and a linear model, u_i = u_i_ref / alpha,
so the posterior SD should fall as 1 / sqrt(N_eff) with

    N_eff(k) = sum_{i<=k} u_i^2 / u_tip^2

read from each run's run_info.json (clean mean u_true). Two truth values are drawn:

    population mean   mean(alpha_i)           -- the quantity of interest
    harmonic mean     1 / mean(1 / alpha_i)   -- what the mean response actually encodes

A mean-only inversion should land on the harmonic mean, not the population mean.

Every alpha is also given as E = alpha * E_ref [GPa], E_ref from run_info.json
(206.9 GPa).

Writes results.csv, u_mean_information.png and u_mean_posterior_alpha.png.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else r"D:\KratosProjects\MCMC\Analysis\sensors_u_mean")

SIGMA = 3.788e-08
E_REF_GPA = 206.9      # fallback when run_info.json has no e_ref

BOX = "#5b7c99"
EDGE = "#1f3b57"
TRUTH = "#d62728"
HARM = "#2ca02c"
PRED = "#2a78d6"
NAIVE = "#d85a30"

FIELDS = ["run", "label", "n_sensors", "stations", "n_eff", "sigma_assumed", "e_ref_GPa",
          "alpha_mean", "alpha_std", "alpha_p2.5", "alpha_p97.5",
          "E_mean_GPa", "E_std_GPa", "E_p2.5_GPa", "E_p97.5_GPa",
          "alpha_pop_mean", "alpha_harmonic_mean", "alpha_ls_from_data",
          "E_pop_mean_GPa", "E_harmonic_mean_GPa", "E_ls_from_data_GPa",
          "bias_pop_pct", "bias_harm_pct", "z_pop", "z_harm",
          "covers_pop", "covers_harm", "sd_ratio", "sd_pred_ratio",
          "n_levels", "n_forward_solves", "wall_time_s", "logcE"]

SKIPPED = []


def posterior_samples(folder):
    for path in sorted(glob.glob(str(folder / "*.npz"))):
        with np.load(path) as z:
            keys = [k for k in z.files if k.startswith("level_")]
            if keys:
                return np.asarray(z[sorted(keys)[-1]])[:, 0]
    return None


def collect(folder):
    summary_file = folder / "summary.json"
    info_file = folder / "run_info.json"
    if not summary_file.exists():
        why = "no run_info.json either -- the sweep never reached this case"
        if info_file.exists():
            code = json.loads(info_file.read_text()).get("returncode")
            why = ("prepared only, Phase 2 not run" if code is None
                   else f"Phase 2 exited with {code}" if code
                   else "Phase 2 reported success but wrote nothing")
        SKIPPED.append((folder.name, why))
        return None

    summary = json.loads(summary_file.read_text())
    info = json.loads(info_file.read_text())
    zone = summary["zones"][0]
    m, s = zone["alpha_mean"], zone["alpha_std"]
    lo, hi = zone["alpha_p2.5"], zone["alpha_p97.5"]
    pop, harm = info["alpha_pop_mean"], info["alpha_harmonic_mean"]
    ls = info.get("alpha_ls_from_data")
    e_ref = info.get("e_ref", E_REF_GPA * 1e9) / 1e9

    return {
        "run": folder.name,
        "label": info["label"],
        "n_sensors": info["n_sensors"],
        "stations": " ".join(format(x, ".1f") for x in info["stations"]),
        "n_eff": info["n_eff"],
        "sigma_assumed": info["sigma_assumed"],
        "e_ref_GPa": e_ref,
        "alpha_mean": m,
        "alpha_std": s,
        "alpha_p2.5": lo,
        "alpha_p97.5": hi,
        "E_mean_GPa": m * e_ref,
        "E_std_GPa": s * e_ref,
        "E_p2.5_GPa": lo * e_ref,
        "E_p97.5_GPa": hi * e_ref,
        "alpha_pop_mean": pop,
        "alpha_harmonic_mean": harm,
        "alpha_ls_from_data": ls,
        "E_pop_mean_GPa": pop * e_ref,
        "E_harmonic_mean_GPa": harm * e_ref,
        "E_ls_from_data_GPa": None if ls is None else ls * e_ref,
        "bias_pop_pct": 100.0 * (m - pop) / pop,
        "bias_harm_pct": 100.0 * (m - harm) / harm,
        "z_pop": (m - pop) / s,
        "z_harm": (m - harm) / s,
        "covers_pop": lo <= pop <= hi,
        "covers_harm": lo <= harm <= hi,
        "n_levels": len(summary["tempering_q"]),
        "n_forward_solves": summary["n_forward_solves"],
        "wall_time_s": info.get("wall_time_s"),
        "logcE": summary["logcE"],
        "_samples": posterior_samples(folder),
    }


def save(fig, out):
    with open(str(out), "wb") as f:
        fig.savefig(f, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_information(rows, out):
    k = np.array([r["n_sensors"] for r in rows], float)
    n_eff = np.array([r["n_eff"] for r in rows])
    sd = np.array([r["alpha_std"] for r in rows])
    sd0, ne0 = sd[0], n_eff[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4))
    fig.subplots_adjust(hspace=0.36)

    ax1.plot(k, k, color=NAIVE, ls="--", lw=1.4, label=r"naive  $N_{eff} = k$")
    ax1.plot(k, n_eff, "o-", color=EDGE, ms=7, lw=1.6,
             label=r"$N_{eff}$ from clean mean $u_{true}$")
    for ki, ne in zip(k, n_eff):
        ax1.annotate(f"{ne:.2f}", (ki, ne), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=EDGE)
    ax1.set_xticks(k)
    ax1.set_xlabel("sensors  k   (tip first, then inward)")
    ax1.set_ylabel(r"effective sensor count  $N_{eff}$")
    ax1.set_title(r"Mean-only: $N_{eff} = \sum u_i^2 / u_{tip}^2$ under one absolute sigma",
                  fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, framealpha=0.9)

    ax2.plot(k, sd0 / np.sqrt(k / k[0]), "s--", color=NAIVE, ms=6, lw=1.4,
             label=r"naive  $SD_1/\sqrt{k}$")
    ax2.plot(k, sd0 / np.sqrt(n_eff / ne0), "^--", color=PRED, ms=7, lw=1.5,
             label=r"predicted  $SD_1/\sqrt{N_{eff}}$")
    ax2.plot(k, sd, "o-", color=EDGE, ms=7, lw=1.8, label="posterior SD (sampled)")
    e_ref = rows[0]["e_ref_GPa"]
    sec = ax2.secondary_yaxis("right", functions=(lambda a: a * e_ref, lambda e: e / e_ref))
    sec.set_ylabel(r"posterior SD of $E$  [GPa]")
    ax2.set_xticks(k)
    ax2.set_xlabel("sensors  k")
    ax2.set_ylabel(r"posterior SD of $\alpha$")
    ax2.set_title("Sampled SD against the Fisher prediction", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, framealpha=0.9)

    pred = sd0 / np.sqrt(n_eff / ne0)
    resid = 100.0 * (sd - pred) / pred
    fig.text(0.5, 0.015,
             f"max deviation from the Fisher prediction: {np.abs(resid).max():.1f}%",
             ha="center", fontsize=8, color="grey", style="italic")
    save(fig, out)


def figure_posterior(rows, out):
    have = all(r["_samples"] is not None for r in rows)
    rng = np.random.default_rng(0)
    data = [r["_samples"] if r["_samples"] is not None
            else rng.normal(r["alpha_mean"], r["alpha_std"], 4000) for r in rows]
    pos = np.arange(len(rows))
    pop, harm = rows[0]["alpha_pop_mean"], rows[0]["alpha_harmonic_mean"]
    e_ref = rows[0]["e_ref_GPa"]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.boxplot(data, positions=pos, widths=0.55, vert=False, patch_artist=True,
               showfliers=False, whis=(2.5, 97.5),
               medianprops=dict(color=EDGE, lw=2),
               boxprops=dict(facecolor=BOX, edgecolor=EDGE, alpha=0.75, lw=1.2),
               whiskerprops=dict(color=EDGE, lw=1.2),
               capprops=dict(color=EDGE, lw=1.2))
    for p, r in zip(pos, rows):
        ax.errorbar(r["alpha_mean"], p + 0.36, xerr=r["alpha_std"], fmt="D", mfc="white",
                    mec=EDGE, ecolor=EDGE, ms=6, capsize=3, zorder=5)
    ax.axvline(pop, color=TRUTH, ls="--", lw=1.8, zorder=1)
    ax.axvline(harm, color=HARM, ls=":", lw=2.0, zorder=1)

    lo = min(min(np.percentile(d, 2.5) for d in data), pop, harm)
    hi = max(max(np.percentile(d, 97.5) for d in data), pop, harm)
    span = hi - lo
    ax.set_xlim(lo - 0.09 * span, hi + 0.78 * span)
    for p, r in zip(pos, rows):
        mark = "" if r["covers_harm"] else "   *"
        ax.text(hi + 0.03 * span, p,
                f"$\\alpha$ {r['alpha_mean']:.4f} $\\pm$ {r['alpha_std']:.4f}\n"
                f"E {r['E_mean_GPa']:.2f} $\\pm$ {r['E_std_GPa']:.2f} GPa\n"
                f"pop {r['bias_pop_pct']:+.2f}%   harm {r['bias_harm_pct']:+.2f}%{mark}",
                va="center", ha="left", fontsize=8.5, color=EDGE, linespacing=1.4)

    ax.set_yticks(pos)
    ax.set_yticklabels([f"{r['n_sensors']} sensor{'s' if r['n_sensors'] > 1 else ''}"
                        f"   $N_{{eff}}$ = {r['n_eff']:.2f}" for r in rows])
    ax.set_ylabel("sensor count")
    ax.set_xlabel(r"$\alpha = E / E_{ref}$")
    sec = ax.secondary_xaxis("top", functions=(lambda a: a * e_ref, lambda e: e / e_ref))
    sec.set_xlabel(f"E  [GPa]   (E_ref = {e_ref:g} GPa)", fontsize=9)
    ax.set_title(f"Mean-only posterior vs. sensor count   |   sigma {SIGMA:.4e} per sensor"
                 f"   |   * = 95% CI misses harmonic mean", fontsize=12, pad=40)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        Line2D([], [], color=TRUTH, ls="--", lw=1.8,
               label=f"population mean  {pop:.5f}  ({pop * e_ref:.2f} GPa)"),
        Line2D([], [], color=HARM, ls=":", lw=2.0,
               label=f"harmonic mean  {harm:.5f}  ({harm * e_ref:.2f} GPa)"),
        Line2D([], [], marker="D", color=EDGE, mfc="white", mec=EDGE, ms=6,
               label=r"posterior mean $\pm$ SD"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9, frameon=False)

    note = ("boxes from posterior samples (2.5-97.5% whiskers)" if have
            else "boxes Gaussian-implied from summary.json (mean, SD)")
    fig.text(0.5, -0.08, note, ha="center", fontsize=8, color="grey", style="italic")
    save(fig, out)


def main():
    if not ARCHIVE.exists():
        sys.exit(f"no such archive: {ARCHIVE}")

    folders = [f for f in sorted(ARCHIVE.iterdir()) if f.is_dir()]
    rows = [r for r in (collect(f) for f in folders) if r]
    if SKIPPED:
        print(f"*** {len(SKIPPED)} of {len(folders)} run folders have no result "
              f"and are NOT in the table below ***")
        for name, why in SKIPPED:
            print(f"    {name:<20} {why}")
        print()
    if not rows:
        sys.exit(f"no summary.json found under {ARCHIVE}")
    rows.sort(key=lambda r: r["n_sensors"])

    sig = {r["sigma_assumed"] for r in rows}
    if len(sig) > 1:
        print("WARNING: sigma_assumed varies across runs -- this sweep should hold it fixed")
    elif abs(sig.pop() - SIGMA) / SIGMA > 1e-3:
        print(f"WARNING: sigma_assumed is not {SIGMA:.4e}")
    if len({(r["alpha_pop_mean"], r["alpha_harmonic_mean"]) for r in rows}) > 1:
        print("WARNING: truth values differ between runs -- mixed datasets?")

    base = rows[0]
    if base["n_sensors"] != 1:
        print("WARNING: no N=1 run -- SD ratios are relative to the smallest run present")
    for r in rows:
        r["sd_ratio"] = r["alpha_std"] / base["alpha_std"]
        r["sd_pred_ratio"] = float(np.sqrt(base["n_eff"] / r["n_eff"]))

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)

    pop, harm = base["alpha_pop_mean"], base["alpha_harmonic_mean"]
    e_ref = base["e_ref_GPa"]
    print(f"E_ref = {e_ref:g} GPa")
    print(f"truth: population mean alpha {pop:.6f} = {pop * e_ref:.2f} GPa   "
          f"harmonic mean alpha {harm:.6f} = {harm * e_ref:.2f} GPa\n")

    header = (f"{'run':<6}{'k':>4}{'N_eff':>8}{'alpha':>9}{'sd':>9}{'95% CI alpha':>20}"
              f"{'E [GPa]':>10}{'sd':>8}{'95% CI [GPa]':>18}")
    print("posterior")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_a = f"[{r['alpha_p2.5']:.4f},{r['alpha_p97.5']:.4f}]"
        ci_e = f"[{r['E_p2.5_GPa']:.2f},{r['E_p97.5_GPa']:.2f}]"
        print(f"{r['label']:<6}{r['n_sensors']:>4}{r['n_eff']:>8.2f}{r['alpha_mean']:>9.4f}"
              f"{r['alpha_std']:>9.4f}{ci_a:>20}{r['E_mean_GPa']:>10.2f}{r['E_std_GPa']:>8.2f}"
              f"{ci_e:>18}")

    header = (f"{'run':<6}{'SD/SD1':>8}{'pred':>7}{'LS alpha':>10}{'LS GPa':>9}"
              f"{'pop%':>8}{'z_pop':>7}{'harm%':>8}{'z_harm':>7}{'cov p/h':>9}"
              f"{'lvls':>5}{'solves':>7}{'time_s':>8}")
    print("\nchecks")
    print(header)
    print("-" * len(header))
    for r in rows:
        ls = "" if r["alpha_ls_from_data"] is None else format(r["alpha_ls_from_data"], ">10.4f")
        ls_e = "" if r["E_ls_from_data_GPa"] is None else format(r["E_ls_from_data_GPa"], ">9.2f")
        t = "" if r["wall_time_s"] is None else format(r["wall_time_s"], ">8.0f")
        cov = f"{'y' if r['covers_pop'] else 'N'}/{'y' if r['covers_harm'] else 'N'}"
        print(f"{r['label']:<6}{r['sd_ratio']:>8.3f}{r['sd_pred_ratio']:>7.3f}{ls:>10}{ls_e:>9}"
              f"{r['bias_pop_pct']:>8.2f}{r['z_pop']:>7.2f}"
              f"{r['bias_harm_pct']:>8.2f}{r['z_harm']:>7.2f}{cov:>9}"
              f"{r['n_levels']:>5}{r['n_forward_solves']:>7}{t}")

    last = rows[-1]
    print(f"\nN_eff reaches {last['n_eff']:.2f} with {last['n_sensors']} sensors "
          f"(naive would be {last['n_sensors']})")
    print(f"SD shrinks {1 / last['sd_ratio']:.2f}x, Fisher predicts "
          f"{1 / last['sd_pred_ratio']:.2f}x, naive 1/sqrt(k) predicts "
          f"{np.sqrt(last['n_sensors'] / base['n_sensors']):.2f}x")
    print(f"coverage: population mean {sum(r['covers_pop'] for r in rows)}/{len(rows)}, "
          f"harmonic mean {sum(r['covers_harm'] for r in rows)}/{len(rows)} "
          f"of the 95% intervals")

    figure_information(rows, ARCHIVE / "u_mean_information.png")
    figure_posterior(rows, ARCHIVE / "u_mean_posterior_alpha.png")
    print(f"\nwrote {out}\n      {ARCHIVE / 'u_mean_information.png'}"
          f"\n      {ARCHIVE / 'u_mean_posterior_alpha.png'}")


if __name__ == "__main__":
    main()
