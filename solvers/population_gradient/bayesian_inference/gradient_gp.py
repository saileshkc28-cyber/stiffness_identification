"""Gradient-enhanced Gaussian process in one input, t = ln(E / e_scale).

One independent GP per sensor, trained on values u(t) AND slopes du/dt.
Matern 5/2 kernel with unit variance, zero mean after normalisation:

    k(r)  = (1 + a|r| + a^2 r^2 / 3) exp(-a|r|)          a = sqrt(5) / ell
    k1(r) = dk/dr   = -(a^2 r / 3) (1 + a|r|) exp(-a|r|)
    k2(r) = d2k/dr2 = -(a^2 / 3) (1 + a|r| - a^2 r^2) exp(-a|r|)

    cov(u(t_i),  u(t_j))  =  k(r_ij)      cov(u(t_i),  u'(t_j)) = -k1(r_ij)
    cov(u'(t_i), u(t_j))  =  k1(r_ij)     cov(u'(t_i), u'(t_j)) = -k2(r_ij)

The length scale ell is set per sensor by maximum marginal likelihood with
the signal variance profiled out. numpy and scipy only, no Kratos.

Self-test:
    python gradient_gp.py
"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize_scalar

JITTER_MAX = 1e-6        # last jitter tried before the Cholesky gives up
N_GRID = 40              # log-spaced ell values in the coarse search
ELL_BOUNDS = (0.05, 10.0)  # ell search range, in units of (t_max - t_min)


class GPCholeskyError(np.linalg.LinAlgError):
    """Cholesky failed even with the largest allowed jitter."""


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
def matern52(r, ell):
    """k(r), unit variance."""
    ar = (np.sqrt(5.0) / ell) * np.abs(r)
    return (1.0 + ar + ar ** 2 / 3.0) * np.exp(-ar)


def matern52_d1(r, ell):
    """k1(r) = dk/dr; odd in r, zero at r = 0."""
    a = np.sqrt(5.0) / ell
    ar = a * np.abs(r)
    return -(a ** 2 * r / 3.0) * (1.0 + ar) * np.exp(-ar)


def matern52_d2(r, ell):
    """k2(r) = d2k/dr2; even in r, -a^2/3 at r = 0."""
    a = np.sqrt(5.0) / ell
    ar = a * np.abs(r)
    return -(a ** 2 / 3.0) * (1.0 + ar - ar ** 2) * np.exp(-ar)


def joint_cov(t1, t2, ell):
    """Covariance of [u(t1); u'(t1)] with [u(t2); u'(t2)], shape (2 n1, 2 n2).

    With t1 = t2 this is the training matrix K. With t1 = test points, its top
    n1 rows are the cross-covariance rows for u(t*) ([k, -k1]) and its bottom
    n1 rows those for u'(t*) ([k1, -k2]).
    """
    R = np.subtract.outer(np.asarray(t1, float), np.asarray(t2, float))  # R[i, j] = t1_i - t2_j
    k1 = matern52_d1(R, ell)
    return np.block([[matern52(R, ell), -k1],
                     [k1, -matern52_d2(R, ell)]])


def cholesky_with_jitter(K, jitter0):
    """Lower Cholesky factor of K + jitter I, raising the jitter x10 up to JITTER_MAX.

    Returns (L, jitter that worked); raises GPCholeskyError if none works.
    """
    eye = np.eye(K.shape[0])
    jitter = float(jitter0)
    while True:
        try:
            return np.linalg.cholesky(K + jitter * eye), jitter
        except np.linalg.LinAlgError:
            if jitter >= JITTER_MAX * (1.0 - 1e-9):  # tolerance: 1e-10 * 10^4 is not exactly 1e-6
                raise GPCholeskyError(
                    f"Cholesky of the {K.shape[0]}x{K.shape[0]} gradient-enhanced covariance "
                    f"failed for every jitter from {jitter0:.0e} to {JITTER_MAX:.0e}")
            jitter *= 10.0


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class GradientEnhancedGP:
    """Values + slopes GP per sensor in t = ln(E / e_scale)."""

    def __init__(self, e_scale_Pa, jitter=1e-10):
        self.e_scale = float(e_scale_Pa)
        self.jitter = float(jitter)

    def t_of(self, e_Pa):
        e = np.asarray(e_Pa, dtype=float).ravel()
        if np.any(~np.isfinite(e)) or np.any(e <= 0.0):
            raise ValueError("E must be finite and > 0")
        return np.log(e / self.e_scale)

    # ---- hyperparameters -------------------------------------------------
    def _objective(self, ell, t, z):
        """2n log(s2_hat) + log det K, with s2_hat = z^T K^-1 z / (2n). +inf if K fails."""
        try:
            L, _ = cholesky_with_jitter(joint_cov(t, t, ell), self.jitter)
        except GPCholeskyError:
            return np.inf
        v = solve_triangular(L, z, lower=True)   # z^T K^-1 z = |L^-1 z|^2
        s2_hat = (v @ v) / z.size
        if not s2_hat > 0.0:
            return np.inf
        return z.size * np.log(s2_hat) + 2.0 * np.sum(np.log(np.diag(L)))

    def _fit_ell(self, t, z):
        """Coarse log-spaced grid over ell, then bounded refinement around the best."""
        span = t.max() - t.min()
        grid = span * np.logspace(np.log10(ELL_BOUNDS[0]), np.log10(ELL_BOUNDS[1]), N_GRID)
        obj = np.array([self._objective(ell, t, z) for ell in grid])
        if not np.any(np.isfinite(obj)):
            raise GPCholeskyError(f"covariance not factorisable for any ell in "
                                  f"[{grid[0]:.3g}, {grid[-1]:.3g}]")
        i = int(np.argmin(obj))

        # refine in log(ell) between the grid neighbours of the best point;
        # Brent's method needs finite values, so a failed factorisation becomes 1e300
        lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, N_GRID - 1)])
        res = minimize_scalar(lambda x: min(self._objective(np.exp(x), t, z), 1e300),
                              bounds=(lo, hi), method="bounded")
        if res.success and res.fun < obj[i]:
            return float(np.exp(res.x))
        return float(grid[i])

    # ---- fit / predict ---------------------------------------------------
    def fit(self, e_train_Pa, u_train_m, dudt_train):
        """u_train_m, dudt_train: shape (n, n_sensors); dudt is du/dt in metres."""
        t = self.t_of(e_train_Pa)
        u = np.asarray(u_train_m, dtype=float).reshape(t.size, -1)
        g = np.asarray(dudt_train, dtype=float).reshape(u.shape)
        if t.size < 2 or np.unique(t).size != t.size:
            raise ValueError("need at least 2 distinct training E values")

        self.t_train_ = t
        n_s = u.shape[1]
        self.mu_, self.sd_ = np.empty(n_s), np.empty(n_s)
        self.ell_, self.s2_, self.jitter_ = np.empty(n_s), np.empty(n_s), np.empty(n_s)
        self.alpha_ = []
        for s in range(n_s):
            # 1. normalise: values shifted and scaled, slopes only scaled
            mu, sd = u[:, s].mean(), u[:, s].std()
            if not sd > 0.0:
                raise ValueError(f"sensor {s}: training values are constant, cannot normalise")
            z = np.concatenate([(u[:, s] - mu) / sd, g[:, s] / sd])

            # 2. length scale by profiled maximum marginal likelihood
            ell = self._fit_ell(t, z)

            # 3. factorise K at that ell, keep the jitter that worked
            L, jit = cholesky_with_jitter(joint_cov(t, t, ell), self.jitter)
            v = solve_triangular(L, z, lower=True)

            # 4. weights alpha = K^-1 z; the posterior mean does not depend on s2
            #    (s2 K* (s2 K)^-1 z), so s2_ is kept for reference only
            self.alpha_.append(cho_solve((L, True), z))
            self.mu_[s], self.sd_[s] = mu, sd
            self.ell_[s], self.s2_[s], self.jitter_[s] = ell, (v @ v) / z.size, jit
        return self

    def _predict_both(self, e_Pa):
        """Posterior means of u and du/dt at e_Pa, each shape (m, n_sensors)."""
        ts = self.t_of(e_Pa)
        m = ts.size
        u = np.empty((m, self.ell_.size))
        dudt = np.empty_like(u)
        for s in range(self.ell_.size):
            mean = joint_cov(ts, self.t_train_, self.ell_[s]) @ self.alpha_[s]
            # undo the normalisation: u = mu + sd * u_norm, du/dt = sd * slope_norm
            u[:, s] = self.mu_[s] + self.sd_[s] * mean[:m]
            dudt[:, s] = self.sd_[s] * mean[m:]
        return u, dudt

    def predict(self, e_Pa):
        """Predicted u, shape (m, n_sensors)."""
        return self._predict_both(e_Pa)[0]

    def predict_dudt(self, e_Pa):
        """Predicted du/dt, shape (m, n_sensors)."""
        return self._predict_both(e_Pa)[1]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.interpolate import CubicHermiteSpline

    # 1. analytic k1, k2 against central differences of k, k1 (test only)
    r = np.array([-2.0, -0.9, -0.25, -1e-3, 0.0, 1e-3, 0.25, 0.9, 2.0])
    h = 1e-6
    for ell in (0.3, 2.0):
        a = np.sqrt(5.0) / ell
        fd1 = (matern52(r + h, ell) - matern52(r - h, ell)) / (2 * h)
        fd2 = (matern52_d1(r + h, ell) - matern52_d1(r - h, ell)) / (2 * h)
        e1 = np.max(np.abs(matern52_d1(r, ell) - fd1))
        e2 = np.max(np.abs(matern52_d2(r, ell) - fd2))
        print(f"check 1, ell = {ell}: max |k1 - FD| = {e1:.2e}, max |k2 - FD| = {e2:.2e}")
        if e1 > 1e-6 * a or e2 > 1e-6 * a ** 2:
            raise AssertionError(f"kernel derivatives disagree with finite differences at ell = {ell}")

    # 2. fit u(t) = -1.9e-6 exp(-t), du/dt = -u, on 4 equally spaced t
    e_scale = 206.9e9
    t_lo, t_hi = np.log(80.0 / 206.9), np.log(400.0 / 206.9)
    t_tr = np.linspace(t_lo, t_hi, 4)
    u_tr = (-1.9e-6 * np.exp(-t_tr))[:, None]
    e_tr = e_scale * np.exp(t_tr)
    gp = GradientEnhancedGP(e_scale).fit(e_tr, u_tr, -u_tr)
    eu = np.max(np.abs(gp.predict(e_tr) - u_tr)) / np.max(np.abs(u_tr))
    eg = np.max(np.abs(gp.predict_dudt(e_tr) + u_tr)) / np.max(np.abs(u_tr))
    print(f"check 2: training reproduction, rel. error values {eu:.2e}, slopes {eg:.2e}")
    # not machine precision: the jitter acts as a tiny nugget
    if eu > 1e-5 or eg > 1e-5:
        raise AssertionError("GP does not reproduce the training values/slopes")

    # 3. dense accuracy against a cubic Hermite spline on the same data
    t_d = np.linspace(t_lo, t_hi, 200)
    u_d = -1.9e-6 * np.exp(-t_d)
    scale = np.max(np.abs(u_d))
    err_gp = np.max(np.abs(gp.predict(e_scale * np.exp(t_d))[:, 0] - u_d)) / scale
    herm = CubicHermiteSpline(t_tr, u_tr[:, 0], -u_tr[:, 0])
    err_h = np.max(np.abs(herm(t_d) - u_d)) / scale
    print(f"check 3: 200 dense points, max error / max|u|: GP+slope {err_gp:.3e}, "
          f"Hermite {err_h:.3e}")
    print(f"         ell_ = {gp.ell_[0]:.4g} (t span {t_hi - t_lo:.4g}), "
          f"jitter_ = {gp.jitter_[0]:.0e}")

    print("all checks passed")
