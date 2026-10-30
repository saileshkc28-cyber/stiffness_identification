"""Turn the three-point results into ONE smooth recovered distribution.

The weighted mixture 1/6 p_1 + 2/3 p_2 + 1/6 p_3 is three narrow spikes, because three
support points are not a continuum. This script recovers a continuous distribution from
the same three inversions, without any extra Kratos solves.

Method: stochastic collocation. The three Gauss-Hermite nodes z = (-sqrt3, 0, +sqrt3)
are collocation points of the underlying standard normal variable. Fit the quadratic
Lagrange interpolant through the three recovered means,

    alpha(z) = sum_j  m_j * L_j(z),        L_j = the quadratic basis on the three nodes,

then draw z ~ N(0,1) continuously and evaluate. This is exact at the nodes, is the
degree-2 polynomial-chaos surrogate implied by a three-point rule, and produces a
smooth distribution rather than three spikes.

Two variants are written:

  population   alpha(z) only -- the recovered POPULATION spread, the Phase 1 target
  predictive   alpha(z) + within-posterior noise s(z) -- population + inference
               uncertainty, the wider of the two

The honest caveat, which belongs in any caption: three points fix a quadratic, so this
recovers the population's mean, spread and skew, but cannot resolve finer structure.
It is a surrogate built from three inversions, not a sampled distribution.

    python smooth_three_point.py
    python smooth_three_point.py output_three_point
    python smooth_three_point.py output_three_point 200000
"""
import json
import os
import sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "output_three_point"
n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
CASE_NAMES = ["point_1_flexible", "point_2_central", "point_3_stiff"]
Z_NODES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
WEIGHTS = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])

# ------------------------------------------------------------------ read the results
cases = []
for name in CASE_NAMES:
    path = os.path.join(root, name, "posterior_summary.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path} -- run the three-point inference first")
    with open(path) as f:
        cases.append(json.load(f))

if any(c.get("status") != "ok" for c in cases):
    raise SystemExit("not all three cases completed; cannot build the surrogate")

means = np.array([c["alpha_posterior_mean"] for c in cases])
sds = np.array([c["alpha_posterior_sd"] for c in cases])
E_ref = float(cases[0]["E_ref"])

print(f"{'case':>18} {'z':>10} {'weight':>9} {'alpha mean':>12} {'alpha sd':>11}")
for c, z, w, m, s in zip(cases, Z_NODES, WEIGHTS, means, sds):
    print(f"{c['case_name']:>18} {z:>+10.6f} {w:>9.4f} {m:>12.6f} {s:>11.6f}")

if not (means[0] < means[1] < means[2]) and not (means[0] > means[1] > means[2]):
    print("\nWARNING: the three recovered means are not monotone in z. The quadratic "
          "surrogate assumes a monotone response and may be unreliable.")


# -------------------------------------------------------- quadratic Lagrange surrogate
def lagrange(z, nodes, values):
    """Degree-2 interpolant through the three (node, value) pairs. Exact at the nodes."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    total = np.zeros_like(z)
    for j in range(len(nodes)):
        basis = np.ones_like(z)
        for k in range(len(nodes)):
            if k != j:
                basis *= (z - nodes[k]) / (nodes[j] - nodes[k])
        total += values[j] * basis
    return total


# the surrogate must reproduce the quadrature moments it was built from
quad_mean = float(np.sum(WEIGHTS * means))
quad_between = float(np.sqrt(np.sum(WEIGHTS * (means - quad_mean) ** 2)))
check = lagrange(Z_NODES, Z_NODES, means)
assert np.allclose(check, means, rtol=1e-12), "interpolant is not exact at the nodes"
print(f"\ninterpolant exact at all three nodes: yes")

rng = np.random.default_rng(20260802)
z = rng.standard_normal(n_draws)

alpha_population = lagrange(z, Z_NODES, means)
sd_of_z = np.clip(lagrange(z, Z_NODES, sds), 1e-12, None)     # sd must stay positive
alpha_predictive = alpha_population + sd_of_z * rng.standard_normal(n_draws)

E_population = E_ref * alpha_population
E_predictive = E_ref * alpha_predictive

# ------------------------------------------------------------------------- statistics
print(f"\n{'quantity':>28} {'mean':>12} {'sd':>12}")
print(f"{'three-point quadrature':>28} {quad_mean:>12.6f} {quad_between:>12.6f}")
print(f"{'collocation population':>28} {alpha_population.mean():>12.6f} "
      f"{alpha_population.std(ddof=1):>12.6f}")
print(f"{'collocation predictive':>28} {alpha_predictive.mean():>12.6f} "
      f"{alpha_predictive.std(ddof=1):>12.6f}")

target = None
combined_file = os.path.join(root, "combined", "combined_summary.json")
if os.path.isfile(combined_file):
    with open(combined_file) as f:
        combined = json.load(f)
    if "phase1_target_alpha_mean" in combined:
        target = (combined["phase1_target_alpha_mean"],
                  combined["phase1_target_alpha_sd"])
        print(f"{'Phase 1 target':>28} {target[0]:>12.6f} {target[1]:>12.6f}")
        print(f"\nrelative error vs target : mean "
              f"{alpha_population.mean() / target[0] - 1:+.2%}   "
              f"sd {alpha_population.std(ddof=1) / target[1] - 1:+.2%}")

out_dir = os.path.join(root, "combined")
os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(os.path.join(out_dir, "smooth_posterior_samples.npz"),
                    z=z, alpha_population=alpha_population,
                    alpha_predictive=alpha_predictive,
                    E_population=E_population, E_predictive=E_predictive,
                    nodes=Z_NODES, node_means=means, node_sds=sds,
                    E_ref=np.array([E_ref]))

summary = {
    "method": "degree-2 stochastic collocation on the three Gauss-Hermite nodes",
    "caveat": ("a quadratic through three points: recovers mean, spread and skew, "
               "not finer shape. A surrogate, not a sampled distribution."),
    "n_draws": int(n_draws),
    "E_ref": E_ref,
    "node_z": Z_NODES.tolist(),
    "node_weights": WEIGHTS.tolist(),
    "node_alpha_means": means.tolist(),
    "node_alpha_sds": sds.tolist(),
    "quadrature_mean_alpha": quad_mean,
    "quadrature_between_case_sd_alpha": quad_between,
    "population_mean_alpha": float(alpha_population.mean()),
    "population_sd_alpha": float(alpha_population.std(ddof=1)),
    "population_mean_E": float(E_population.mean()),
    "population_sd_E": float(E_population.std(ddof=1)),
    "predictive_mean_alpha": float(alpha_predictive.mean()),
    "predictive_sd_alpha": float(alpha_predictive.std(ddof=1)),
    "predictive_mean_E": float(E_predictive.mean()),
    "predictive_sd_E": float(E_predictive.std(ddof=1)),
    "population_alpha_ci_2_5": float(np.percentile(alpha_population, 2.5)),
    "population_alpha_ci_97_5": float(np.percentile(alpha_population, 97.5)),
}
if target:
    summary["phase1_target_alpha_mean"] = target[0]
    summary["phase1_target_alpha_sd"] = target[1]
with open(os.path.join(out_dir, "smooth_posterior_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------------------------ plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not available; data written, plot skipped")

NAVY = "#1a3a5a"
MAIN = matplotlib.colors.to_rgba("#4878a8")
colors = plt.cm.viridis(np.linspace(0.15, 0.9, 3))


def normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def gpa_axis(ax):
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda x: x * E_ref / 1e9, lambda x: x * 1e9 / E_ref))
    secondary.set_xlabel("E [GPa]", fontsize=9)


# ============================== figure 1: the recovered distribution, plot_posterior idiom
mean_p = float(alpha_population.mean())
std_p = float(alpha_population.std(ddof=1))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(alpha_population, bins=80, density=True, color=MAIN, edgecolor="white",
        label=f"recovered population (n = {n_draws})")
ax.axvline(mean_p, color=NAVY, lw=1.8,
           label=rf"mean $\alpha$ = {mean_p:.4f} $\pm$ {std_p:.4f}")
for s in (mean_p - std_p, mean_p + std_p):
    ax.axvline(s, color=NAVY, lw=1.0, ls=":")
if target:
    grid = np.linspace(*ax.get_xlim(), 600)
    ax.plot(grid, normal_pdf(grid, target[0], target[1]), color="crimson", ls="--",
            lw=1.5, label=rf"Phase 1 target $\mathcal{{N}}$({target[0]:.3f}, "
                          rf"{target[1]:.3f})")
ax.set_xlabel(r"$\alpha = E/E_{ref}$")
ax.set_ylabel("recovered density")
ax.legend(frameon=False, fontsize=9)
gpa_axis(ax)
fig.savefig(os.path.join(out_dir, "smooth_posterior_alpha.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {os.path.join(out_dir, 'smooth_posterior_alpha.png')}")

# ================================ figure 2: the surrogate that produced it, same palette
fig, ax = plt.subplots(figsize=(7, 4.2))
grid = np.linspace(-3.5, 3.5, 400)
ax.plot(grid, lagrange(grid, Z_NODES, means), color=NAVY, lw=1.8,
        label=r"quadratic $\alpha(z)$")
for zj, mj, sj, w, color in zip(Z_NODES, means, sds, WEIGHTS, colors):
    ax.errorbar(zj, mj, yerr=1.96 * sj, fmt="o", color=color, ms=9, capsize=4,
                lw=1.4, zorder=3,
                label=rf"$z$ = {zj:+.3f}, $w$ = {w:.4f}, $\alpha$ = {mj:.4f}")
ax.set_xlabel(r"$z \sim \mathcal{N}(0,1)$")
ax.set_ylabel(r"recovered $\alpha$")
ax.set_title("collocation surrogate through the three nodes\n"
             "bars = 95% within-posterior interval", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(out_dir, "smooth_surrogate.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {os.path.join(out_dir, 'smooth_surrogate.png')}")

print(f"wrote {os.path.join(out_dir, 'smooth_posterior_samples.npz')}")
print(f"wrote {os.path.join(out_dir, 'smooth_posterior_summary.json')}")

print(f"\n{'quantity':>10} {'mean':>12} {'sd':>12} {'cov':>8} "
      f"{'2.5%':>12} {'97.5%':>12}")
for label, data in (("alpha", alpha_population), ("E [GPa]", E_population / 1e9)):
    m, s = data.mean(), data.std(ddof=1)
    print(f"{label:>10} {m:>12.6f} {s:>12.6f} {s / abs(m):>8.4f} "
          f"{np.percentile(data, 2.5):>12.6f} {np.percentile(data, 97.5):>12.6f}")
