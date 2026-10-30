"""Hierarchical likelihood for a population of nominally identical specimens.

Each specimen i has its own stiffness E_i, drawn from p(E | eta). E_i is never
estimated: it is integrated out on the fixed quadrature grid, leaving only the
population parameters eta to infer.

    p(u_i | eta) = INT p(u_i | E) p(E | eta) dE

p(u_i | E) is evaluated once for every (specimen, node) pair and cached, so a
proposal costs one population density evaluation and one logsumexp -- no
forward solves at all.
"""

import csv

import numpy as np
import KratosMultiphysics as Kratos
from scipy.special import logsumexp

from population_model import FAMILIES
from quadrature import LogEGrid
from response_surrogate import ResponseSurrogate

_LOG2PI = np.log(2.0 * np.pi)


def read_observations(path, require_valid=True, sensor_names=None, n_outputs=None):
    """Observation-only reader. Returns the u_hat columns and nothing else --
    alpha_true, E_true and xi are ground truth and must not reach inference.

    With sensor_names (the surrogate's outputs, in its order) exactly the columns
    u_hat_<name> are taken, in that order. Without them the file order is kept and
    only accepted when the column count equals n_outputs; the caller warns."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path} is empty")

    if sensor_names:
        hat_cols = [f"u_hat_{name}" for name in sensor_names]
        missing = [name for name, c in zip(sensor_names, hat_cols) if c not in rows[0]]
        if missing:
            raise RuntimeError(f"{path} has no u_hat_ column for the surrogate's "
                               f"sensor(s) {', '.join(missing)}")
    else:
        hat_cols = [c for c in rows[0] if c.startswith("u_hat_")]
        if not hat_cols:
            raise RuntimeError(f"no u_hat_* columns in {path}")
        if n_outputs is not None and len(hat_cols) != n_outputs:
            raise RuntimeError(
                f"{path} has {len(hat_cols)} u_hat_ column(s) but the surrogate "
                f"returns {n_outputs} output(s), and it records no sensor names to "
                "select them by")

    if require_valid:
        if "valid" not in rows[0]:
            raise RuntimeError(
                f"{path} has no 'valid' column -- run clean_phase1.py first")
        rows = [r for r in rows if int(r["valid"]) == 1]

    obs = np.array([[float(r[c]) for c in hat_cols] for r in rows])
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("non-finite observation in the valid rows")
    return obs, [c.replace("u_hat_", "") for c in hat_cols]


class PopulationLikelihood:
    """Likelihood only. The prior lives in the sampler's ERADist object."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "observations_file"   : "../damaged_system/phase1_distribution_runs/phase1_samples_clean.csv",
            "require_valid_column": true,
            "surrogate_file"      : "surrogate/response_surrogate.joblib",
            "population_family"   : "normal",
            "quadrature"          : {
                "n_nodes"   : 512,
                "e_min_Pa"  : 80.0e9,
                "e_max_Pa"  : 400.0e9
            },
            "noise_model"         : {
                "type"  : "gaussian_iid",
                "sigma" : 0.0
            }
        }""")

    def __init__(self, settings):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters())
        settings["quadrature"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["quadrature"])
        settings["noise_model"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["noise_model"])

        if settings["noise_model"]["type"].GetString() != "gaussian_iid":
            raise RuntimeError("unsupported noise model")
        self.sigma = settings["noise_model"]["sigma"].GetDouble()
        if not self.sigma > 0.0:
            raise RuntimeError("noise_model.sigma must be positive and is an "
                               "analyst assumption -- type it in, do not read "
                               "it from Phase 1")

        # the surrogate first: its sensor names decide which columns are read
        self.surrogate = ResponseSurrogate.load(settings["surrogate_file"].GetString())
        names = self.surrogate.sensor_names
        if not names:
            Kratos.Logger.PrintWarning(
                "PopulationLikelihood",
                "the surrogate records no sensor names; the observation columns are "
                "taken in file order and matched to the surrogate outputs by position "
                "only. Rebuild the surrogate to match them by name.")
        self.obs, self.sensors = read_observations(
            settings["observations_file"].GetString(),
            settings["require_valid_column"].GetBool(),
            sensor_names=names, n_outputs=self.surrogate.n_sensors_)
        Kratos.Logger.PrintInfo(
            "PopulationLikelihood", "observation columns, in surrogate output order: "
            + ", ".join(f"u_hat_{s}" for s in self.sensors))

        q = settings["quadrature"]
        self.grid = LogEGrid(q["e_min_Pa"].GetDouble(), q["e_max_Pa"].GetDouble(),
                             q["n_nodes"].GetInt(), self.surrogate.e_scale)

        family = settings["population_family"].GetString()
        if family not in FAMILIES:
            raise RuntimeError(f"unknown population family '{family}'")
        self.population = FAMILIES[family](self.grid.e_min, self.grid.e_max)
        self.family = family

        self._build_table()
        self.n_calls = 0

    # ---------------------------------------------------------------- table
    def _build_table(self):
        """cond_loglik[i, q] = log p(u_i | E_q). Built once; the data and the
        surrogate are both frozen, so no proposal can change it."""
        G = self.surrogate.predict(self.grid.E)            # (Q, S)
        if G.shape[1] != self.obs.shape[1]:
            raise RuntimeError(
                f"surrogate returns {G.shape[1]} sensor(s) but the observations "
                f"have {self.obs.shape[1]}")
        n_s = self.obs.shape[1]
        resid = self.obs[:, None, :] - G[None, :, :]       # (N, Q, S)
        self.cond_loglik = (-0.5 * n_s * (_LOG2PI + 2.0 * np.log(self.sigma))
                            - 0.5 * np.einsum("nqs,nqs->nq", resid, resid)
                            / self.sigma ** 2)
        self.table_bytes = self.cond_loglik.nbytes

    # ----------------------------------------------------------- likelihood
    def LogLikelihood(self, eta):
        log_p = self.population.log_density(self.grid.E, np.atleast_1d(eta))
        if log_p is None:
            return -np.inf
        terms = self.cond_loglik + (self.grid.log_weight + log_p)[None, :]
        total = float(logsumexp(terms, axis=1).sum())
        return total if np.isfinite(total) else -np.inf

    def __call__(self, eta):
        """Scalar float: numpy >= 2 rejects length-1 arrays inside SMC_aCS."""
        self.n_calls += 1
        return self.LogLikelihood(eta)

    # ------------------------------------------------------------ reporting
    def Describe(self):
        return {
            "n_observations": int(self.obs.shape[0]),
            "n_sensors": int(self.obs.shape[1]),
            "sensors": self.sensors,
            "sigma_assumed_m": self.sigma,
            "population_family": self.family,
            "quadrature": self.grid.spacing_report(),
            "table_shape": list(self.cond_loglik.shape),
            "table_MiB": round(self.table_bytes / 1024 ** 2, 3),
            "surrogate_identity": self.surrogate.identity_,
        }
