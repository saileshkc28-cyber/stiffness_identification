"""Post-processing only. Reads the saved posterior and, if available, the Phase 1
truth record. Kept out of MainBayesian.py so the inference never touches the truth.

Writes two figures:
  output/posterior_alpha.png   the q=1 posterior on its own
  output/posterior_levels.png  one panel per SMC level, plus a box plot
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


def draw_level(ax, a, mean, std, color, full=False):
    ax.hist(a, bins=40, density=True, color=color, edgecolor="white")
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
draw_level(ax, alpha, mean_l[-1], std_l[-1], colors[-1], full=True)
ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=9)
gpa_axis(ax)
fig.savefig("output/posterior_alpha.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote output/posterior_alpha.png")

# ================================================ figure 2: every level + box plot
ncol = 2
nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(7.0 * ncol, 4.3 * nrow + 4.2))
gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [1.0],
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

# ------------------------------------------------ bottom: box plot across levels
ax = fig.add_subplot(gs[nrow, :])
bp = ax.boxplot(levels, vert=True, widths=0.55, patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#1a3a5a", lw=1.6),
                whiskerprops=dict(color="#555555"),
                capprops=dict(color="#555555"),
                boxprops=dict(edgecolor="#555555"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)

x = np.arange(1, n + 1)
ax.plot(x, mean_l, marker="o", ms=5, color="#1a3a5a", lw=1.2, zorder=3,
        label=r"mean $\alpha$")
if truth is not None:
    ax.axhline(truth, color="crimson", ls="--", lw=1.5,
               label=rf"$\alpha_{{true}}$ = {truth:.4f}")

for xi, s, hi in zip(x, std_l, [a.max() for a in levels]):
    ax.annotate(rf"$\sigma$ = {s:.4f}", (xi, hi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#555555")

ax.set_xticks(x)
ax.set_xticklabels([f"level {i}\nq = {qi:.4f}" for i, qi in enumerate(q)], fontsize=8)
ax.set_ylabel(r"$\alpha = E/E_{ref}$")
ax.set_title(rf"posterior contracts {std_l[0] / std_l[-1]:.0f}$\times$ from prior to $q=1$",
             fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(True, axis="y", alpha=0.25)

secondary = ax.secondary_yaxis(
    "right", functions=(lambda y: y * E_ref / 1e9, lambda y: y * 1e9 / E_ref))
secondary.set_ylabel("E [GPa]", fontsize=9)

fig.savefig("output/posterior_levels.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote output/posterior_levels.png\n")

print(f"{'level':>5} {'q':>8} {'mean':>9} {'std':>9} {'cov':>7} {'unique':>7} {'range':>9}")
for i, (a, qi) in enumerate(zip(levels, q)):
    print(f"{i:>5} {qi:>8.4f} {mean_l[i]:>9.4f} {std_l[i]:>9.4f} "
          f"{std_l[i] / abs(mean_l[i]):>7.3f} {np.unique(a).size:>7d} "
          f"{a.max() - a.min():>9.4f}")
