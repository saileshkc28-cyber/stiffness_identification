"""Vector-valued response surrogate u(E) for the beam/plate forward model."""

import hashlib
import json
import platform

import joblib
import numpy as np
import scipy
import sklearn
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

_TINY = 1e-300


class DomainError(ValueError):
    pass


class ResponseSurrogate:
    """One GP per sensor, fitted in t = log(E / E_scale), predicting the clean
    sensor vector. Never extrapolates: out-of-domain input raises DomainError."""

    def __init__(self, e_scale_Pa, e_min_Pa, e_max_Pa, gp_jitter=1e-10,
                 n_restarts=8, sensor_names=None, sensor_data_file=None,
                 random_state=20260911):
        if not (0.0 < e_min_Pa < e_max_Pa):
            raise ValueError("require 0 < e_min_Pa < e_max_Pa")
        self.e_scale = float(e_scale_Pa)
        self.e_min = float(e_min_Pa)
        self.e_max = float(e_max_Pa)
        self.gp_jitter = float(gp_jitter)
        self.n_restarts = int(n_restarts)
        # output columns in sensor-file order; None for surrogates that predate this
        self.sensor_names = list(sensor_names) if sensor_names else None
        self.sensor_data_file = str(sensor_data_file) if sensor_data_file else None
        # one seed for every sensor's GP, so each GP depends only on its own column
        self.random_state = int(random_state)
        self.gps_ = None
        self.identity_ = None

    # ---- domain -----------------------------------------------------------
    def t_of(self, e_Pa):
        e = np.asarray(e_Pa, dtype=float).ravel()
        if np.any(~np.isfinite(e)) or np.any(e <= 0.0):
            raise DomainError("E must be finite and positive")
        return np.log(e / self.e_scale)

    def check_domain(self, e_Pa):
        e = np.asarray(e_Pa, dtype=float).ravel()
        lo = e < self.e_min * (1.0 - 1e-12)
        hi = e > self.e_max * (1.0 + 1e-12)
        if np.any(lo | hi):
            bad = e[lo | hi]
            raise DomainError(
                "E outside validated surrogate domain "
                f"[{self.e_min:.6e}, {self.e_max:.6e}] Pa; "
                f"{bad.size} offending value(s), first {bad[0]:.6e}. "
                "Expand and revalidate offline, then restart inference."
            )

    # ---- fit / predict ----------------------------------------------------
    def fit(self, e_train_Pa, u_train_m):
        e = np.asarray(e_train_Pa, dtype=float).ravel()
        u = np.atleast_2d(np.asarray(u_train_m, dtype=float))
        if u.shape[0] != e.size:
            u = u.T
        if u.shape[0] != e.size:
            raise ValueError("u_train_m must have one row per training E")
        if self.sensor_names is not None and len(self.sensor_names) != u.shape[1]:
            raise ValueError(f"{len(self.sensor_names)} sensor names for "
                             f"{u.shape[1]} response column(s)")
        self.check_domain(e)

        t = self.t_of(e).reshape(-1, 1)
        # per column, not along axis 0: numpy sums a 2-D axis in a layout-dependent
        # order, which would make a sensor's GP depend on how many columns sit beside it
        self.y_mean_ = np.array([u[:, s].mean() for s in range(u.shape[1])])
        std = np.array([u[:, s].std() for s in range(u.shape[1])])
        std[std < _TINY] = 1.0
        self.y_std_ = std
        z = (u - self.y_mean_) / self.y_std_

        span = float(t.max() - t.min())
        self.gps_ = []
        for s in range(u.shape[1]):
            kernel = ConstantKernel(1.0, (1e-6, 1e8)) * Matern(
                length_scale=0.5 * span,
                length_scale_bounds=(1e-2 * span, 1e2 * span),
                nu=2.5,
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.gp_jitter,
                normalize_y=False,
                n_restarts_optimizer=self.n_restarts,
                random_state=self.random_state,
            )
            gp.fit(t, z[:, s])
            self.gps_.append(gp)

        self.n_sensors_ = u.shape[1]
        self.e_train_ = e
        self.identity_ = self._identity(e, u)
        return self

    def predict(self, e_Pa, return_std=False):
        if self.gps_ is None:
            raise RuntimeError("surrogate is not fitted")
        e = np.asarray(e_Pa, dtype=float).ravel()
        self.check_domain(e)
        t = self.t_of(e).reshape(-1, 1)

        mean = np.empty((e.size, self.n_sensors_))
        sd = np.empty_like(mean)
        for s, gp in enumerate(self.gps_):
            m, d = gp.predict(t, return_std=True)
            mean[:, s] = m * self.y_std_[s] + self.y_mean_[s]
            sd[:, s] = d * self.y_std_[s]
        return (mean, sd) if return_std else mean

    # ---- identity / persistence ------------------------------------------
    def _identity(self, e, u):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(e, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(u, dtype=np.float64).tobytes())
        h.update(f"{self.e_scale}|{self.e_min}|{self.e_max}|{self.gp_jitter}".encode())
        return {
            "training_hash": h.hexdigest()[:16],
            "n_training": int(e.size),
            "n_sensors": int(u.shape[1]),
            "e_scale_Pa": self.e_scale,
            "e_min_Pa": self.e_min,
            "e_max_Pa": self.e_max,
            "gp_jitter": self.gp_jitter,
            "random_state": self.random_state,
            "sensor_names": self.sensor_names,
            "sensor_data_file": self.sensor_data_file,
            "kernels": [str(g.kernel_) for g in self.gps_],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
            },
        }

    def save(self, path):
        joblib.dump(self, path)
        with open(str(path) + ".json", "w") as f:
            json.dump(self.identity_, f, indent=2)

    @staticmethod
    def load(path):
        surrogate = joblib.load(path)
        # surrogates saved before these attributes existed
        for name in ("sensor_names", "sensor_data_file", "random_state"):
            if not hasattr(surrogate, name):
                setattr(surrogate, name, None)
        return surrogate


class InterpBaseline:
    """Monotone cubic interpolant in t. Accuracy/cost baseline for the pilot."""

    def __init__(self, e_scale_Pa):
        self.e_scale = float(e_scale_Pa)

    def fit(self, e_train_Pa, u_train_m):
        e = np.asarray(e_train_Pa, dtype=float).ravel()
        u = np.atleast_2d(np.asarray(u_train_m, dtype=float))
        if u.shape[0] != e.size:
            u = u.T
        order = np.argsort(e)
        t = np.log(e[order] / self.e_scale)
        self.interps_ = [PchipInterpolator(t, u[order, s]) for s in range(u.shape[1])]
        return self

    def predict(self, e_Pa):
        t = np.log(np.asarray(e_Pa, dtype=float).ravel() / self.e_scale)
        return np.column_stack([f(t) for f in self.interps_])


def error_report(u_pred, u_exact, sigma_noise_m):
    """Signed / RMSE / max errors and error-to-noise ratios."""
    p = np.atleast_2d(u_pred)
    x = np.atleast_2d(u_exact)
    sig = np.asarray(sigma_noise_m, dtype=float).ravel()
    if sig.size == 1:
        sig = np.full(p.shape[1], sig[0])
    err = p - x
    rel = np.abs(err) / sig
    return {
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error_m": float(np.max(np.abs(err))),
        "mean_signed_error_m": float(np.mean(err)),
        "max_error_over_sigma": float(np.max(rel)),
        "rmse_over_sigma": float(np.sqrt(np.mean(rel ** 2))),
        "per_sensor_max_over_sigma": np.max(rel, axis=0).tolist(),
    }
