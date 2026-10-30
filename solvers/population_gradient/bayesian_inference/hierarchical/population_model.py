"""Population densities p(E | eta), evaluated at fixed quadrature nodes.

Each model is renormalised over the surrogate domain, so the density integrates
to one on [e_min, e_max] and a proposal that pushes mass outside the domain is
penalised rather than silently ignored.
"""

import numpy as np
from scipy.special import log_ndtr

_LOG2PI = np.log(2.0 * np.pi)


class NormalPopulation:
    """E ~ Normal(mu, sd), truncated to the surrogate domain.
    Matches a generator of the form E = E_ref * (1 + c * xi), xi ~ N(0, 1)."""

    n_params = 2
    names = ("mu_E", "sd_E")

    def __init__(self, e_min_Pa, e_max_Pa):
        self.e_min = float(e_min_Pa)
        self.e_max = float(e_max_Pa)

    def log_density(self, E, eta):
        mu, sd = float(eta[0]), float(eta[1])
        if not (sd > 0.0):
            return None
        z = (np.asarray(E, float) - mu) / sd
        log_mass = _log_diff_ndtr((self.e_max - mu) / sd, (self.e_min - mu) / sd)
        if not np.isfinite(log_mass):
            return None
        return -0.5 * (_LOG2PI + z * z) - np.log(sd) - log_mass

    def moments(self, eta):
        return float(eta[0]), float(eta[1])


class LogNormalPopulation:
    """log E ~ Normal(m, s), truncated to the surrogate domain."""

    n_params = 2
    names = ("log_median_E", "sd_logE")

    def __init__(self, e_min_Pa, e_max_Pa):
        self.e_min = float(e_min_Pa)
        self.e_max = float(e_max_Pa)

    def log_density(self, E, eta):
        m, s = float(eta[0]), float(eta[1])
        if not (s > 0.0):
            return None
        lE = np.log(np.asarray(E, float))
        z = (lE - m) / s
        log_mass = _log_diff_ndtr((np.log(self.e_max) - m) / s,
                                  (np.log(self.e_min) - m) / s)
        if not np.isfinite(log_mass):
            return None
        return -0.5 * (_LOG2PI + z * z) - np.log(s) - lE - log_mass

    def moments(self, eta):
        m, s = float(eta[0]), float(eta[1])
        mean = np.exp(m + 0.5 * s * s)
        return mean, mean * np.sqrt(np.expm1(s * s))


def _log_diff_ndtr(hi, lo):
    """log(Phi(hi) - Phi(lo)), stable in both tails."""
    if hi < lo:
        return -np.inf
    if hi + lo > 0.0:                      # work in the upper tail instead
        hi, lo = -lo, -hi
    a, b = log_ndtr(hi), log_ndtr(lo)
    d = b - a
    if d >= 0.0:
        return -np.inf
    return a + np.log(-np.expm1(d))


FAMILIES = {"normal": NormalPopulation, "lognormal": LogNormalPopulation}
