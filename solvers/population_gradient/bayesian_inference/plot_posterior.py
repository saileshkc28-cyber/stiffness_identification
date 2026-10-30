"""Post-processing only. Reads the saved posterior and, if available, the Phase 1
truth record. Kept out of MainBayesian.py so the inference never touches the truth.

Writes two figures:
  output/posterior_alpha.png   the q=1 posterior on its own
  output/posterior_levels.png  one panel per SMC level, plus convergence
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("output/smc_levels.npz")
q = data["q"]
E_ref = float(np.atleast_1d(data["E_ref"])[0])
levels = [data[f"level_{i:02d}"][:, 0] for i in range(len(q))]
alpha = levels[-1]

truth = None
truth_file = "../damaged_system/StructuralMaterials.json"
if os.path.exists(truth_file):
    pid = int(data["prop_ids"][0])
    for block in json.load(open(truth_file))["properties"]:
        if block["properties_id"] == pid:
            truth = block["Material"]["Variables"]["YOUNG_MODULUS"] / E_ref

n = len(levels)
colors = plt.cm.viridis(np.linspace(0.15, 0.9, n))
colors[-1] = matplotlib.colors.to_rgba("#4878a8")   # q=1 keeps the posterior blue
mean_l = np.array([a.mean() for a in levels])
std_l = np.array([a.std(ddof=1) for a in levels])


def gpa_axis(ax):
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda x: x * E_ref / 1e9, lambda x: x * 1e9 / E_ref))
    secondary.set_xlabel("E [GPa]", fontsize=9)


def draw_level(ax, a, mean, std, color, label_n=True):
    ax.hist(a, bins=40, density=True, color=color, edgecolor="white",
            label=(f"n = {len(a)}, unique = {np.unique(a).size}" if label_n
                   else f"posterior samples (n={len(a)})"))
    ax.axvline(mean, color="#1a3a5a", lw=1.8,
               label=rf"mean $\alpha$ = {mean:.4f} $\pm$ {std:.4f}")
    for s in (mean - std, mean + std):
        ax.axvline(s, color="#1a3a5a", lw=1.0, ls=":")
    if truth is not None:
        ax.axvline(truth, color="crimson", ls="--", lw=1.5,
                   label=rf"$\alpha_{{true}}$ = {truth:.4f}")
    ax.set_xlabel(r"$\alpha = E/E_{ref}$")


# ===================================================== figure 1: the posterior
fig, ax = plt.subplots(figsize=(7, 4.2))
draw_level(ax, alpha, mean_l[-1], std_l[-1], colors[-1], label_n=False)
ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=9)
gpa_axis(ax)
fig.savefig("output/posterior_alpha.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote output/posterior_alpha.png")

# ============================================ figure 2: every level + convergence
ncol = 2
nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(7.0 * ncol, 4.3 * nrow + 4.0))
gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.92],
                      hspace=0.75, wspace=0.22)

for i, (a, qi, c) in enumerate(zip(levels, q, colors)):
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    draw_level(ax, a, mean_l[i], std_l[i], c)

    lo, hi = a.min(), a.max()
    pad = 0.04 * (hi - lo)
    if truth is not None and not lo - pad <= truth <= hi + pad:
        ax.set_xlim(min(lo - pad, truth - pad), max(hi + pad, truth + pad))
    else:
        ax.set_xlim(lo - pad, hi + pad)

    ax.set_ylabel("density")
    ax.set_title(rf"level {i}   $q$ = {qi:.4f}", fontsize=11, pad=32)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    gpa_axis(ax)

x = np.arange(n)
labels = [f"{qi:.4f}" for qi in q]

ax = fig.add_subplot(gs[nrow, 0])
ax.errorbar(x, mean_l, yerr=std_l, marker="o", color="#1a3a5a", capsize=4, lw=1.5,
            label=r"mean $\pm$ 1 std")
if truth is not None:
    ax.axhline(truth, color="crimson", ls="--", lw=1.5, label=r"$\alpha_{true}$")
for xi, m, c in zip(x, mean_l, colors):
    ax.plot(xi, m, marker="o", color=c, ms=7, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_xlabel(r"tempering parameter $q$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("convergence of the mean", fontsize=11)
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[nrow, 1])
ax.semilogy(x, std_l, marker="o", color="#1a3a5a", lw=1.5)
for xi, s, c in zip(x, std_l, colors):
    ax.plot(xi, s, marker="o", color=c, ms=7, zorder=3)
    ax.annotate(f"{s:.4f}", (xi, s), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_xlabel(r"tempering parameter $q$")
ax.set_ylabel(r"posterior std of $\alpha$")
ax.set_title(rf"spread contracts {std_l[0] / std_l[-1]:.0f}$\times$", fontsize=11)
ax.grid(True, which="both", axis="y", alpha=0.25)

fig.savefig("output/posterior_levels.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote output/posterior_levels.png\n")

print(f"{'level':>5} {'q':>8} {'mean':>9} {'std':>9} {'cov':>7} {'unique':>7} {'range':>9}")
for i, (a, qi) in enumerate(zip(levels, q)):
    print(f"{i:>5} {qi:>8.4f} {mean_l[i]:>9.4f} {std_l[i]:>9.4f} "
          f"{std_l[i] / abs(mean_l[i]):>7.3f} {np.unique(a).size:>7d} "
          f"{a.max() - a.min():>9.4f}")