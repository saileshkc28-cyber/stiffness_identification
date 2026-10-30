"""Post-processing only. Same spirit as plot_posterior.py, but one figure with two
stacked panels sharing the alpha axis:

  top     the q=1 posterior  (histogram + KDE, mean, +/-1 sd, 95% CI band, truth)
  bottom  a horizontal box plot of the same samples, whiskers at the 2.5/97.5
          percentiles so the box and the shaded CI above line up exactly

Usage
  python plot_posterior_box.py                          # defaults to point_2_central
  python plot_posterior_box.py <case_folder> [outfile]

Reads posterior_samples.npz (falls back to smc_levels.npz) from the case folder.
Truth alpha, if it exists, comes from three_point_inputs/<case>/metadata.json or
from output_three_point/combined/case_summary.csv. Never touched by the inference.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
except ImportError:                                        # scipy is optional
    gaussian_kde = None

BLUE = "#4878a8"
DARK = "#1a3a5a"
RED = "crimson"

DEFAULT_CASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "output_three_point", "point_2_central")


# ------------------------------------------------------------------ loading
def load_samples(case_dir):
    """alpha samples, E_ref, case name. posterior_samples.npz first."""
    f = os.path.join(case_dir, "posterior_samples.npz")
    if os.path.exists(f):
        d = np.load(f)
        alpha = np.ravel(d["alpha"])
        E_ref = float(np.atleast_1d(d["E_ref"])[0])
    else:
        f = os.path.join(case_dir, "smc_levels.npz")
        d = np.load(f)
        alpha = np.ravel(d[f"level_{len(d['q']) - 1:02d}"])
        E_ref = float(np.atleast_1d(d["E_ref"])[0])
    return alpha, E_ref, os.path.basename(os.path.normpath(case_dir))


def load_truth(case_dir, case):
    """alpha_true from the case metadata, else from the combined case_summary."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(case_dir)))

    meta = os.path.join(root, "three_point_inputs", case, "metadata.json")
    if os.path.exists(meta):
        with open(meta) as fh:
            v = json.load(fh).get("alpha_true")
        if v is not None:
            return float(v)

    csv_file = os.path.join(root, "output_three_point", "combined", "case_summary.csv")
    if os.path.exists(csv_file):
        with open(csv_file, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("case_name") == case and row.get("alpha_true"):
                    return float(row["alpha_true"])
    return None


# -------------------------------------------------------------------- stats
def summarise(alpha):
    return dict(
        n=len(alpha),
        mean=float(alpha.mean()),
        std=float(alpha.std(ddof=1)),
        median=float(np.median(alpha)),
        q1=float(np.percentile(alpha, 25)),
        q3=float(np.percentile(alpha, 75)),
        lo=float(np.percentile(alpha, 2.5)),
        hi=float(np.percentile(alpha, 97.5)),
    )


# --------------------------------------------------------------------- plot
def plot(case_dir=DEFAULT_CASE, outfile=None):
    alpha, E_ref, case = load_samples(case_dir)
    truth = load_truth(case_dir, case)
    s = summarise(alpha)
    outfile = outfile or os.path.join(case_dir, "posterior_box.png")

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.4, 5.2), sharex=True,
        gridspec_kw=dict(height_ratios=[4.0, 1.0], hspace=0.10))

    # ---- top: posterior -------------------------------------------------
    ax.hist(alpha, bins=40, density=True, color=BLUE, edgecolor="white",
            label=f"posterior (n = {s['n']})")

    if gaussian_kde is not None:
        grid = np.linspace(alpha.min(), alpha.max(), 400)
        ax.plot(grid, gaussian_kde(alpha)(grid), color=DARK, lw=1.4, alpha=0.85)

    ax.axvspan(s["lo"], s["hi"], color=BLUE, alpha=0.16, zorder=0)
    ax.axvline(s["mean"], color=DARK, lw=1.8, label="mean")
    for v in (s["mean"] - s["std"], s["mean"] + s["std"]):
        ax.axvline(v, color=DARK, lw=1.0, ls=":")
    if truth is not None:
        ax.axvline(truth, color=RED, ls="--", lw=1.5, label="true")

    ax.set_ylabel("posterior density")
    ax.set_title(rf"{case}   posterior of $\alpha = E/E_{{ref}}$", fontsize=11, pad=30)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    note = rf"mean {s['mean']:.4f} $\pm$ {s['std']:.4f}" + "\n" + \
           f"95% CI [{s['lo']:.4f}, {s['hi']:.4f}]"
    if truth is not None:
        note += "\n" + rf"$\alpha_{{true}}$ {truth:.4f}"
    ax.text(0.015, 0.97, note, transform=ax.transAxes, fontsize=8, color=DARK,
            va="top", ha="left", linespacing=1.5)

    secondary = ax.secondary_xaxis(
        "top", functions=(lambda x: x * E_ref / 1e9, lambda x: x * 1e9 / E_ref))
    secondary.set_xlabel("E [GPa]", fontsize=9)

    # ---- bottom: box plot ----------------------------------------------
    bp = bx.boxplot(alpha, vert=False, widths=0.55, whis=(2.5, 97.5),
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor=BLUE, edgecolor=DARK, lw=1.2),
                    medianprops=dict(color="white", lw=2.0),
                    whiskerprops=dict(color=DARK, lw=1.2),
                    capprops=dict(color=DARK, lw=1.2),
                    flierprops=dict(marker="o", ms=2.5, mfc=DARK, mec="none",
                                    alpha=0.35))
    bx.plot(s["mean"], 1, marker="D", ms=5, color=DARK, mec="white", mew=0.8,
            zorder=5)
    if truth is not None:
        bx.axvline(truth, color=RED, ls="--", lw=1.5)

    bx.set_yticks([])
    bx.set_ylim(0.5, 1.62)
    bx.set_xlabel(r"$\alpha = E/E_{ref}$")
    bx.text(0.005, 0.99,
            f"median {s['median']:.4f}   IQR [{s['q1']:.4f}, {s['q3']:.4f}]   "
            "whiskers 2.5 / 97.5 pct",
            transform=bx.transAxes, fontsize=7.5, color=DARK, va="top")
    for side in ("left", "right", "top"):
        bx.spines[side].set_visible(False)

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {outfile}\n")
    print(f"case            {case}")
    print(f"samples         {s['n']}")
    print(f"mean            {s['mean']:.6f}")
    print(f"std             {s['std']:.6f}")
    print(f"median          {s['median']:.6f}")
    print(f"IQR             [{s['q1']:.6f}, {s['q3']:.6f}]")
    print(f"95% CI          [{s['lo']:.6f}, {s['hi']:.6f}]")
    if truth is not None:
        print(f"alpha_true      {truth:.6f}")
        print(f"bias            {s['mean'] - truth:+.6f}"
              f"  ({(s['mean'] - truth) / truth * 100:+.3f} %)")
        print(f"z = bias/std    {(s['mean'] - truth) / s['std']:+.3f}")
        print(f"truth in 95% CI {'yes' if s['lo'] <= truth <= s['hi'] else 'NO'}")
    print(f"E mean          {s['mean'] * E_ref / 1e9:.3f} GPa"
          f"   (E_ref = {E_ref / 1e9:.1f} GPa)")
    return outfile


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE,
         sys.argv[2] if len(sys.argv) > 2 else None)
