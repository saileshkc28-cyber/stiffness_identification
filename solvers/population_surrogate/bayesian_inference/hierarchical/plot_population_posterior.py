"""Post-processing only. Reads the saved population posterior and the Phase 1
truth record. Kept out of MainHierarchical.py so the inference never sees truth.

  output_hierarchical/population_posterior.png   q=1 posterior, both parameters
  output_hierarchical/population_levels.png      box plot per level + convergence
  output_hierarchical/population_final_box.png   the q=1 box plot on its own
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output_hierarchical"
TRUTH_FILE = ("../../damaged_system/phase1_distribution_runs/"
              "phase1_clean_summary.json")
GPA = 1e9
MEAN_C = "#1a3a5a"
FINAL_C = "#4878a8"

data = np.load(os.path.join(OUT, "posterior.npz"), allow_pickle=True)
q = np.asarray(data["q"]).ravel()
names = [str(x) for x in data["names"]]
levels = [data[f"level_{i}"] for i in range(len(q))]
post = data["posterior"]
n = len(levels)

truth = None
if os.path.exists(TRUTH_FILE):
    t = json.load(open(TRUTH_FILE))["population_truth"]
    truth = {"mu_E": t["E_mean_GPa"], "sd_E": t["E_sd_GPa"]}

E_REF = 206.9
sfile = os.path.join(OUT, "summary.json")
if os.path.exists(sfile):
    E_REF = (json.load(open(sfile))["data"]["surrogate_identity"]["e_scale_Pa"]
             / GPA)


def alpha_axis(ax, where="top", label=r"$\alpha = E/E_{ref}$"):
    """Secondary scale in alpha. Every axis in this file is already in GPa.
    Ticks are given enough decimals to stay distinct -- mu_alpha spans about
    0.003, so the default two-decimal formatter would print 1.00 five times."""
    fwd, inv = (lambda v: v / E_REF), (lambda v: v * E_REF)
    horizontal = where in ("top", "bottom")
    sec = (ax.secondary_xaxis(where, functions=(fwd, inv)) if horizontal
           else ax.secondary_yaxis(where, functions=(fwd, inv)))
    lo, hi = (ax.get_xlim() if horizontal else ax.get_ylim())
    span = abs(hi - lo) / E_REF
    dec = 2 if span <= 0 else int(np.clip(np.ceil(-np.log10(span / 5.0)) + 1, 2, 6))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:.{dec}f}")
    (sec.xaxis if horizontal else sec.yaxis).set_major_formatter(fmt)
    (sec.xaxis if horizontal else sec.yaxis).set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=6))
    sec.set_xlabel(label, fontsize=9) if horizontal else sec.set_ylabel(label,
                                                                       fontsize=9)
    return sec


colors = plt.cm.viridis(np.linspace(0.15, 0.9, n))
colors[-1] = matplotlib.colors.to_rgba(FINAL_C)

units = [(l / GPA) for l in levels]           # GPa throughout
post_g = post / GPA
mean_l = np.array([u.mean(axis=0) for u in units])      # (n_levels, 2)
std_l = np.array([u.std(axis=0, ddof=1) for u in units])
labels = {"mu_E": r"$\mu_E$ [GPa]", "sd_E": r"$\sigma_E$ [GPa]"}


def truth_of(k):
    return None if truth is None else truth.get(names[k])


# ================================================ figure 1: the q=1 posterior
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

for k, ax in enumerate(axes[0]):
    v = post_g[:, k]
    ax.hist(v, bins=40, density=True, color=FINAL_C, edgecolor="white",
            label=f"posterior samples (n={len(v)})")
    ax.axvline(v.mean(), color=MEAN_C, lw=1.8,
               label=f"mean = {v.mean():.4f} $\\pm$ {v.std(ddof=1):.4f}")
    for s in (v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1)):
        ax.axvline(s, color=MEAN_C, lw=1.0, ls=":")
    tv = truth_of(k)
    if tv is not None:
        ax.axvline(tv, color="crimson", ls="--", lw=1.5, label=f"truth = {tv:.4f}")
    ax.set_xlabel(labels.get(names[k], names[k]))
    ax.set_ylabel("posterior density")
    ax.legend(frameon=False, fontsize=9)
    alpha_axis(ax, "top", r"$\mu_\alpha$" if k == 0 else r"$\sigma_\alpha$")

ax = axes[1, 0]
ax.plot(post_g[:, 0], post_g[:, 1], ".", ms=2.5, alpha=0.35, color=FINAL_C)
if truth is not None:
    ax.plot(truth["mu_E"], truth["sd_E"], "x", color="crimson", ms=11, mew=2.2,
            label="truth")
    ax.legend(frameon=False, fontsize=9)
r = np.corrcoef(post_g[:, 0], post_g[:, 1])[0, 1]
ax.set_xlabel(labels.get(names[0], names[0]))
ax.set_ylabel(labels.get(names[1], names[1]))
ax.set_title(f"joint posterior   corr = {r:+.3f}", fontsize=11)
ax.grid(alpha=0.25)
alpha_axis(ax, "top", r"$\mu_\alpha$")
alpha_axis(ax, "right", r"$\sigma_\alpha$")

ax = axes[1, 1]
e = np.linspace(max(post_g[:, 0].mean() - 5 * post_g[:, 1].mean(), 1.0),
                post_g[:, 0].mean() + 5 * post_g[:, 1].mean(), 400)
idx = np.random.default_rng(0).choice(len(post_g), size=min(200, len(post_g)),
                                      replace=False)
for i in idx:
    m, s = post_g[i]
    ax.plot(e, np.exp(-0.5 * ((e - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
            color=FINAL_C, alpha=0.04, lw=1)
m, s = post_g[:, 0].mean(), post_g[:, 1].mean()
ax.plot(e, np.exp(-0.5 * ((e - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
        color=MEAN_C, lw=2.0, label="posterior mean")
if truth is not None:
    m, s = truth["mu_E"], truth["sd_E"]
    ax.plot(e, np.exp(-0.5 * ((e - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
            color="crimson", ls="--", lw=1.8, label="Phase 1 truth")
ax.set_xlabel("$E$ [GPa]")
ax.set_ylabel("population density")
ax.set_title("inferred stiffness population", fontsize=11)
ax.legend(frameon=False, fontsize=9)
alpha_axis(ax)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "population_posterior.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/population_posterior.png")

# ================================= figure 2: box plot per level + convergence
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
x = np.arange(n)
tick = [f"{qi:.4f}" for qi in q]

for k in range(2):
    ax = axes[0, k]
    bp = ax.boxplot([u[:, k] for u in units], positions=x, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=MEAN_C, lw=1.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    tv = truth_of(k)
    if tv is not None:
        ax.axhline(tv, color="crimson", ls="--", lw=1.5, label=f"truth = {tv:.4f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tick, rotation=45, fontsize=8)
    ax.set_xlabel(r"tempering parameter $q$")
    ax.set_ylabel(labels.get(names[k], names[k]))
    ax.set_title(f"{names[k]} across SMC levels", fontsize=11, pad=26)
    ax.grid(alpha=0.25, axis="y")
    alpha_axis(ax, "right", r"$\mu_\alpha$" if k == 0 else r"$\sigma_\alpha$")

ax = axes[1, 0]
for k, mk in enumerate(("o", "s")):
    ax.errorbar(x, mean_l[:, k], yerr=std_l[:, k], marker=mk, capsize=4, lw=1.5,
                label=labels.get(names[k], names[k]))
    tv = truth_of(k)
    if tv is not None:
        ax.axhline(tv, color="crimson", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(tick, rotation=45, fontsize=8)
ax.set_xlabel(r"tempering parameter $q$")
ax.set_ylabel("GPa")
ax.set_title("convergence of the means", fontsize=11)
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
for k, mk in enumerate(("o", "s")):
    ax.semilogy(x, std_l[:, k], marker=mk, lw=1.5,
                label=f"{names[k]}  contracts {std_l[0, k] / std_l[-1, k]:.0f}$\\times$")
ax.set_xticks(x)
ax.set_xticklabels(tick, rotation=45, fontsize=8)
ax.set_xlabel(r"tempering parameter $q$")
ax.set_ylabel("posterior std [GPa]")
ax.set_title("spread contraction", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "population_levels.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/population_levels.png")

# ============================ figure 3: both parameters in one box plot
# Different units (206 vs 20 GPa) cannot share an axis, so each box is drawn as
# an offset from the Phase 1 truth. Then both sit on the same GPa scale.
fig, ax = plt.subplots(figsize=(9.0, 3.6))
if truth is not None:
    dev = [post_g[:, k] - truth_of(k) for k in range(2)]
    bp = ax.boxplot(dev, positions=[1, 0], widths=0.5, vert=False,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=MEAN_C, lw=1.8))
    for b in bp["boxes"]:
        b.set_facecolor(FINAL_C)
        b.set_alpha(0.85)
    ax.axvline(0.0, color="crimson", ls="--", lw=1.6, label="Phase 1 truth")
    ax.set_yticks([1, 0])
    ax.set_yticklabels(
        [rf"$\mu_E$" "\n" f"{truth['mu_E']:.3f} GPa\n"
         rf"$\mu_\alpha$ = {truth['mu_E'] / E_REF:.4f}",
         rf"$\sigma_E$" "\n" f"{truth['sd_E']:.3f} GPa\n"
         rf"$\sigma_\alpha$ = {truth['sd_E'] / E_REF:.4f}"])
    ax.set_xlabel("posterior $-$ truth  [GPa]")
    ax.set_title("population parameters at $q$ = 1", fontsize=11, pad=26)
    alpha_axis(ax, "top", r"posterior $-$ truth  [$\alpha$]")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "population_final_box.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/population_final_box.png")

# ==================== figure 4: three design cases for the stiffness population
# Each case picks a (mu, sd) pair from the posterior and draws the population of
# E it implies. Percentiles are chosen so that "worse" always means lower mean
# stiffness together with wider scatter.
CASES = [
    ("central",      50.0, 50.0),
    ("conservative", 16.0, 84.0),
    ("extreme",       2.5, 97.5),
]
Z25, Z75, Z05, Z95 = -0.67449, 0.67449, -1.64485, 1.64485

fig, ax = plt.subplots(figsize=(9.0, 4.0))
stats, rows = [], []
for name, p_mu, p_sd in CASES:
    m = np.percentile(post_g[:, 0], p_mu)
    s = np.percentile(post_g[:, 1], p_sd)
    stats.append({"label": f"{name}\n$\\mu$={m:.2f}  $\\sigma$={s:.2f} GPa\n"
                           f"$\\mu_\\alpha$={m / E_REF:.4f}  "
                           f"$\\sigma_\\alpha$={s / E_REF:.4f}",
                  "med": m, "q1": m + Z25 * s, "q3": m + Z75 * s,
                  "whislo": m + Z05 * s, "whishi": m + Z95 * s, "fliers": []})
    rows.append((name, p_mu, p_sd, m, s, m + Z05 * s))

bp = ax.bxp(stats, positions=[2, 1, 0], widths=0.55, vert=False,
            patch_artist=True, showfliers=False,
            medianprops=dict(color=MEAN_C, lw=1.8))
for b, c in zip(bp["boxes"], (FINAL_C, "#c98b3a", "#b0503f")):
    b.set_facecolor(c)
    b.set_alpha(0.85)
if truth is not None:
    ax.axvline(truth["mu_E"], color="crimson", ls="--", lw=1.5,
               label=f"truth mean = {truth['mu_E']:.2f} GPa")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_xlabel("$E$ [GPa]   (box = central 50% of specimens, whiskers = 5th/95th)")
ax.set_title("stiffness population under three choices of $(\\mu_E, \\sigma_E)$",
             fontsize=11, pad=26)
ax.grid(alpha=0.25, axis="x")
alpha_axis(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "population_design_cases.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/population_design_cases.png\n")

print(f"{'case':>13}{'pct mu':>9}{'pct sd':>9}{'mu [GPa]':>11}{'sd [GPa]':>11}"
      f"{'5th pct E':>12}")
for name, p_mu, p_sd, m, s, lo in rows:
    print(f"{name:>13}{p_mu:>9.1f}{p_sd:>9.1f}{m:>11.3f}{s:>11.3f}{lo:>12.3f}")
print()

# ----------------------------------------------------------------- summary
hdr = f"{'level':>5} {'q':>8}"
for nm in names:
    hdr += f" {nm + ' mean':>13} {nm + ' std':>12}"
print(hdr)
for i, qi in enumerate(q):
    line = f"{i:>5} {qi:>8.4f}"
    for k in range(len(names)):
        line += f" {mean_l[i, k]:>13.4f} {std_l[i, k]:>12.4f}"
    print(line)
if truth is not None:
    print(f"\ntruth   mu_E = {truth['mu_E']:.4f} GPa   sd_E = {truth['sd_E']:.4f} GPa")
    print(f"        mu_alpha = {truth['mu_E'] / E_REF:.6f}   "
          f"sd_alpha = {truth['sd_E'] / E_REF:.6f}")
    print(f"posterior  mu_alpha = {mean_l[-1, 0] / E_REF:.6f} "
          f"+- {std_l[-1, 0] / E_REF:.6f}   "
          f"sd_alpha = {mean_l[-1, 1] / E_REF:.6f} +- {std_l[-1, 1] / E_REF:.6f}")
    for k, nm in enumerate(names):
        tv = truth_of(k)
        if tv is not None:
            z = (mean_l[-1, k] - tv) / std_l[-1, k]
            print(f"  {nm:<6} offset {mean_l[-1, k] - tv:+.4f} GPa  "
                  f"= {z:+.2f} posterior sd")