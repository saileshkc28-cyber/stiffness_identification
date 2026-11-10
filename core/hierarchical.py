"""Hierarchical population inference for several zones, using the additive surrogate.

The population: each zone's alpha ~ Normal(mu_z, sd_z), independent between zones and
truncated to the surrogate range. Six unknowns for three zones. Each specimen's own
alpha is never estimated; it is integrated out:

    p(u_i | eta) = integral of p(u_i | alpha) p(alpha | eta) d alpha

The integral is done on a small grid around each specimen:
1. Best fit of every specimen with the surrogate (Gauss-Newton, no Kratos solves).
2. A local bell shape from the curvature at that fit, and a Gauss-Hermite grid of
   nodes_per_zone points per zone placed on it (7 x 7 x 7 = 343 points for 3 zones).
3. p(u_i | alpha) at every grid point, once. The table does not depend on the
   population, so the sampler only reweights it: zero Kratos solves while sampling.

Run from the case's bayesian_inference folder. Settings come from the "hierarchical"
block of ../case.json; the truth in ../damaged_system/population_summary.json is read
only at the very end, for comparison.
"""
import csv
import importlib
import itertools
import json
import os

import joblib
import numpy as np
import KratosMultiphysics as Kratos
from scipy.special import logsumexp, ndtr

from era.Distributions.ERADist import ERADist
from era.Distributions.ERANataf import ERANataf

OUTPUT = os.path.join("output", "hierarchical")
LOG2PI = np.log(2.0 * np.pi)

with open("../case.json") as f:
    cfg = json.load(f)["hierarchical"]
with open("../noise_sigma.json") as f:
    sigma = json.load(f)["sigma"]
surrogate_file = os.path.join("output", "surrogate", "surrogate.joblib")
if not os.path.exists(surrogate_file):
    raise RuntimeError(f"no surrogate at {surrogate_file}: run the surrogate stage first")
surrogate = joblib.load(surrogate_file)
n_zones = len(surrogate.zone_names)
lo = np.array([c.e_min / c.e_scale for c in surrogate.curves])
hi = np.array([c.e_max / c.e_scale for c in surrogate.curves])
inside = 1e-6 * (hi - lo)

# observations only: the u_hat columns, in the surrogate's sensor order
with open(cfg["observations_file"], newline="") as f:
    rows = list(csv.DictReader(f))
Y = np.array([[float(r[f"u_hat_{n}"]) for n in surrogate.sensor_names] for r in rows])
n_spec, n_sens = Y.shape


def predict_or_nan(alpha):
    """Surrogate readings; NaN rows where alpha lies outside the surrogate range."""
    a = np.atleast_2d(alpha)
    ok = np.all((a >= lo - 1e-12) & (a <= hi + 1e-12), axis=1)
    out = np.full((a.shape[0], n_sens), np.nan)
    if ok.any():
        out[ok] = surrogate.predict(a[ok])
    return out


# 1. best fit of every specimen, all at once (Gauss-Newton with the surrogate)
alpha = np.tile(surrogate.anchor, (n_spec, 1))
h = 1e-4
for iteration in range(50):
    u = surrogate.predict(alpha)
    J = np.empty((n_spec, n_sens, n_zones))
    for z in range(n_zones):
        step = np.where(alpha[:, z] + h > hi[z], -h, h)
        shifted = alpha.copy()
        shifted[:, z] += step
        J[:, :, z] = (surrogate.predict(shifted) - u) / step[:, None]
    JTJ = np.einsum("nsz,nsy->nzy", J, J)
    JTr = np.einsum("nsz,ns->nz", J, Y - u)
    delta = np.linalg.solve(JTJ, JTr[..., None])[..., 0]
    alpha = np.clip(alpha + delta, lo + inside, hi - inside)
    if np.abs(delta).max() < 1e-10:
        break
cov = np.linalg.inv(JTJ) * sigma ** 2            # local bell shape per specimen
chol = np.linalg.cholesky(cov)
fit_sd = np.sqrt(np.einsum("nzz->nz", cov))
at_edge = np.any((alpha <= lo + 2 * inside) | (alpha >= hi - 2 * inside), axis=1)

# 2. Gauss-Hermite grid on each specimen's bell shape
m = cfg["nodes_per_zone"]
x1, w1 = np.polynomial.hermite_e.hermegauss(m)   # for weight exp(-x^2 / 2)
w1 = w1 / np.sqrt(2.0 * np.pi)
Zg = np.array(list(itertools.product(x1, repeat=n_zones)))          # (K, n_zones)
log_W = np.log(np.array([np.prod(w) for w in itertools.product(w1, repeat=n_zones)]))
s = cfg["widen"]
nodes = alpha[:, None, :] + s * np.einsum("nzy,ky->nkz", chol, Zg)  # (N, K, n_zones)
log_q = (-0.5 * np.sum(Zg ** 2, axis=1)[None, :] - 0.5 * n_zones * LOG2PI
         - n_zones * np.log(s) - np.sum(np.log(np.einsum("nzz->nz", chol)), axis=1)[:, None])

# 3. p(u_i | alpha) at every node, once
K = Zg.shape[0]
G = predict_or_nan(nodes.reshape(-1, n_zones)).reshape(n_spec, K, n_sens)
resid = Y[:, None, :] - G
log_lik = -0.5 * np.sum(resid ** 2, axis=2) / sigma ** 2 - 0.5 * n_sens * (LOG2PI + 2 * np.log(sigma))
table = log_W[None, :] + log_lik - log_q
outside = np.isnan(table)
table[outside] = -np.inf
print(f"{n_spec} specimens, {n_sens} sensors, {n_zones} zones; best fits in {iteration + 1} "
      f"Gauss-Newton steps; {int(at_edge.sum())} best fits at the surrogate edge")
print(f"grid: {m} points per zone, {K} per specimen, table {table.shape}, "
      f"{outside.mean() * 100:.2f}% of grid points outside the surrogate range")

# 4. population likelihood, called by the sampler
n_calls = 0


def population_loglik(theta):
    global n_calls
    n_calls += 1
    theta = np.asarray(theta, float).ravel()
    mu, sd = theta[0::2], theta[1::2]
    if np.any(sd <= 0.0):
        return -np.inf
    zs = (nodes - mu) / sd                                        # (N, K, n_zones)
    mass = ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd)
    if np.any(mass <= 0.0):
        return -np.inf
    log_p = np.sum(-0.5 * zs ** 2 - np.log(sd) - 0.5 * LOG2PI - np.log(mass), axis=2)
    total = float(np.sum(logsumexp(table + log_p, axis=1)))
    return total if np.isfinite(total) else -np.inf


class Prior:
    def __init__(self, bounds):
        self.marginals = [ERADist("uniform", "PAR", list(b)) for b in bounds]
        self.dim = len(bounds)

    def EraObject(self):
        return ERANataf(self.marginals, np.identity(self.dim))


names, bounds = [], []
for z in range(n_zones):
    mu_bounds = cfg.get("mu_prior", [[lo[z], hi[z]]] * n_zones)[z]
    names += [f"mu_{z + 1}", f"sd_{z + 1}"]
    bounds += [list(mu_bounds), list(cfg["sd_prior"])]
prior = Prior(bounds)
sampler_cfg = cfg["sampler"]
module = importlib.import_module("samplers." + sampler_cfg["python_module"])
sampler = module.Factory(population_loglik, prior,
                         Kratos.Parameters(json.dumps(sampler_cfg["Parameters"])))
print("sampling the population parameters ...")
sampler.Run()
post = sampler.levels[-1]

# 5. results
os.makedirs(OUTPUT, exist_ok=True)
stats = {}
for k, name in enumerate(names):
    v = post[:, k]
    stats[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                   "p2.5": float(np.percentile(v, 2.5)),
                   "p97.5": float(np.percentile(v, 97.5))}
naive = {"mean_of_best_fits": alpha.mean(axis=0).tolist(),
         "sd_of_best_fits": alpha.std(axis=0, ddof=1).tolist(),
         "typical_fit_sd": np.median(fit_sd, axis=0).tolist()}
summary = {"parameters": names, "posterior": stats, "best_fit_spread": naive,
           "nodes_per_zone": m, "widen": s, "grid_points_outside": float(outside.mean()),
           "best_fits_at_edge": int(at_edge.sum()),
           "logcE": float(sampler.Evidence()),
           "tempering_q": np.asarray(sampler.q).ravel().tolist(),
           "n_likelihood_calls": n_calls, "n_forward_solves": 0,
           "n_specimens": n_spec, "sigma": sigma}
truth_file = "../damaged_system/population_summary.json"
truth = None
if os.path.exists(truth_file):
    with open(truth_file) as f:
        truth = json.load(f)
    summary["truth_for_comparison"] = {
        "alpha_realised_mean": truth["alpha_realised_mean"],
        "alpha_realised_std": truth["alpha_realised_std"]}
with open(os.path.join(OUTPUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
np.savez(os.path.join(OUTPUT, "posterior.npz"), posterior=post, names=np.array(names),
         q=np.asarray(sampler.q),
         **{f"level_{i:02d}": lv for i, lv in enumerate(sampler.levels)})
with open(os.path.join(OUTPUT, "best_fits.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"alpha_{z + 1}" for z in range(n_zones)]
                    + [f"sd_{z + 1}" for z in range(n_zones)])
    for a, d in zip(alpha, fit_sd):
        writer.writerow([repr(float(v)) for v in list(a) + list(d)])

print(f"\n{len(sampler.q)} levels, {n_calls} likelihood calls, 0 Kratos solves, "
      f"logcE = {sampler.Evidence():.3f}")
for z in range(n_zones):
    mu, sd = stats[f"mu_{z + 1}"], stats[f"sd_{z + 1}"]
    line = (f"zone {z + 1}: mean {mu['mean']:.4f} +/- {mu['sd']:.4f}   "
            f"spread {sd['mean']:.4f} +/- {sd['sd']:.4f}")
    if truth:
        line += (f"   | drawn: mean {truth['alpha_realised_mean'][z]:.4f}, "
                 f"spread {truth['alpha_realised_std'][z]:.4f}")
    print(line)
print("spread of the best fits (no hierarchy, inflated by noise):",
      np.array2string(np.array(naive["sd_of_best_fits"]), precision=4))
print(f"written: {OUTPUT}")
