"""Additive response surrogate: one curve per zone, plus optional pair corrections.

    u(alpha) = u(c) + sum over zones of d_z(alpha_z) + sum over pairs of f_ij(alpha_i, alpha_j)
    d_z(a)   = u(c with zone z set to a) - u(c)      (zone z changed alone)
    f_ij     = what the two single curves miss when zones i and j change together

c is the anchor, a centre for the population such as the mean-only result.

Each d_z is fitted with the existing one-dimensional GP class (ResponseSurrogate, one
GP per sensor). The curves alone are exact when every zone's force follows from statics
alone (a statically determinate structure). Zones that touch can still interact at
their shared boundary; f_ij, fitted with a two-dimensional GP per sensor, captures that
for neighbouring pairs. The check against full Kratos runs in build_surrogate.py reports
how well the result matches.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class PairCurve:
    """Interaction of zones i and j: one two-dimensional GP per sensor over
    (alpha_i, alpha_j), scaled to the unit square. Never extrapolates."""

    def __init__(self, i, j, lo, hi, gp_jitter=1e-10, n_restarts=8, random_state=0):
        self.i, self.j = int(i), int(j)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.gp_jitter = float(gp_jitter)
        self.n_restarts = int(n_restarts)
        self.random_state = int(random_state)
        self.gps_ = None

    def _scaled(self, a):
        x = (np.atleast_2d(np.asarray(a, float)) - self.lo) / (self.hi - self.lo)
        if np.any(x < -1e-9) or np.any(x > 1.0 + 1e-9):
            raise ValueError(f"alpha outside the pair ({self.i + 1}, {self.j + 1}) range "
                             f"{self.lo} to {self.hi}")
        return x

    def fit(self, a, y):
        x = self._scaled(a)
        y = np.atleast_2d(np.asarray(y, float))
        self.mean_ = np.array([y[:, s].mean() for s in range(y.shape[1])])
        std = np.array([y[:, s].std() for s in range(y.shape[1])])
        std[std < 1e-300] = 1.0
        self.std_ = std
        z = (y - self.mean_) / self.std_
        self.gps_ = []
        for s in range(y.shape[1]):
            kernel = ConstantKernel(1.0, (1e-6, 1e8)) * Matern(
                length_scale=[0.5, 0.5], length_scale_bounds=(1e-2, 1e2), nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=self.gp_jitter,
                                          normalize_y=False,
                                          n_restarts_optimizer=self.n_restarts,
                                          random_state=self.random_state)
            gp.fit(x, z[:, s])
            self.gps_.append(gp)
        return self

    def predict(self, a):
        x = self._scaled(a)
        z = np.column_stack([gp.predict(x) for gp in self.gps_])
        return z * self.std_ + self.mean_


class AdditiveSurrogate:
    def __init__(self, u_base, e_refs, curves, sensor_names, zone_names, anchor,
                 pairs=()):
        self.u_base = np.asarray(u_base, float)
        self.e_refs = np.asarray(e_refs, float)
        self.curves = list(curves)          # one ResponseSurrogate per zone, input E in Pa
        self.sensor_names = list(sensor_names)
        self.zone_names = list(zone_names)
        self.anchor = np.asarray(anchor, float)
        self.pairs = list(pairs)            # PairCurve objects, possibly none

    def predict(self, alpha):
        """alpha with shape (n_points, n_zones) -> readings (n_points, n_sensors).
        Raises an error outside the fitted range: the surrogate never extrapolates."""
        a = np.atleast_2d(np.asarray(alpha, float))
        if a.shape[1] != len(self.curves):
            raise ValueError(f"expected {len(self.curves)} zones, got {a.shape[1]}")
        u = np.tile(self.u_base, (a.shape[0], 1))
        for z, curve in enumerate(self.curves):
            u += curve.predict(a[:, z] * self.e_refs[z])
        for pair in self.pairs:
            u += pair.predict(a[:, [pair.i, pair.j]])
        return u
