"""Response surrogate u(E) fitted on values AND adjoint slopes du/dt.

A drop-in ResponseSurrogate: PopulationLikelihood loads it with
ResponseSurrogate.load and calls predict(E) exactly as before. Inside, one
gradient_gp.GradientEnhancedGP (Matern 5/2, one GP per sensor) in
t = ln(E / e_scale). Domain check, t_of, save and load are inherited.

numpy / scipy only at fit and predict time, no Kratos. Importable from
bayesian_inference/, which MainHierarchical puts on sys.path, so joblib can
unpickle it there.
"""

import hashlib
import platform

import numpy as np
import scipy
import sklearn

from gradient_gp import GradientEnhancedGP
from response_surrogate import ResponseSurrogate


class GradientResponseSurrogate(ResponseSurrogate):
    """Values + slopes GP per sensor. Never extrapolates: out-of-domain input
    raises DomainError, as in ResponseSurrogate. No predictive std."""

    def __init__(self, e_scale_Pa, e_min_Pa, e_max_Pa, gp_jitter=1e-10,
                 sensor_names=None, sensor_data_file=None, training_csv=None):
        super().__init__(e_scale_Pa, e_min_Pa, e_max_Pa, gp_jitter=gp_jitter,
                         n_restarts=0, sensor_names=sensor_names,
                         sensor_data_file=sensor_data_file)
        # the ell search is a fixed grid + bounded Brent: no restarts, no seed
        self.random_state = None
        self.training_csv = str(training_csv) if training_csv else None
        self.gp_ = None

    # ---- fit / predict ----------------------------------------------------
    def fit(self, e_train_Pa, u_train_m, dudt_train, sensor_names=None,
            sensor_data_file=None, training_csv=None):
        """u_train_m, dudt_train: one row per training E (transposed input is
        accepted, as in ResponseSurrogate.fit). dudt is du/dt in metres.
        The keyword arguments, when given, replace the constructor's values."""
        if sensor_names is not None:
            self.sensor_names = list(sensor_names)
        if sensor_data_file is not None:
            self.sensor_data_file = str(sensor_data_file)
        if training_csv is not None:
            self.training_csv = str(training_csv)

        e = np.asarray(e_train_Pa, dtype=float).ravel()
        u = np.atleast_2d(np.asarray(u_train_m, dtype=float))
        g = np.atleast_2d(np.asarray(dudt_train, dtype=float))
        if u.shape[0] != e.size:
            u = u.T
        if g.shape[0] != e.size:
            g = g.T
        if u.shape[0] != e.size:
            raise ValueError("u_train_m must have one row per training E")
        if g.shape != u.shape:
            raise ValueError(f"dudt_train shape {g.shape} != u_train_m shape {u.shape}")
        if self.sensor_names is not None and len(self.sensor_names) != u.shape[1]:
            raise ValueError(f"{len(self.sensor_names)} sensor names for "
                             f"{u.shape[1]} response column(s)")
        self.check_domain(e)

        self.gp_ = GradientEnhancedGP(self.e_scale, jitter=self.gp_jitter).fit(e, u, g)

        self.n_sensors_ = u.shape[1]
        self.e_train_ = e
        self.identity_ = self._identity(e, u, g)
        return self

    def predict(self, e_Pa):
        """Predicted u, shape (m, n_sensors)."""
        if self.gp_ is None:
            raise RuntimeError("surrogate is not fitted")
        e = np.asarray(e_Pa, dtype=float).ravel()
        self.check_domain(e)
        return self.gp_.predict(e)

    # ---- identity ---------------------------------------------------------
    def _identity(self, e, u, g):
        # same fields as ResponseSurrogate._identity; the slopes enter the hash too
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(e, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(u, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(g, dtype=np.float64).tobytes())
        h.update(f"{self.e_scale}|{self.e_min}|{self.e_max}|{self.gp_jitter}".encode())
        gp = self.gp_
        return {
            "kind": "gradient_enhanced_gp",
            "training_hash": h.hexdigest()[:16],
            "training_csv": self.training_csv,
            "n_training": int(e.size),
            "n_sensors": int(u.shape[1]),
            "e_scale_Pa": self.e_scale,
            "e_min_Pa": self.e_min,
            "e_max_Pa": self.e_max,
            "gp_jitter": self.gp_jitter,
            "random_state": self.random_state,
            "sensor_names": self.sensor_names,
            "sensor_data_file": self.sensor_data_file,
            "kernels": [f"Matern52(ell={gp.ell_[s]:.6g}) on u and du/dt, s2={gp.s2_[s]:.6g}"
                        for s in range(gp.ell_.size)],
            # per sensor: fitted length scale in t, and the jitter the Cholesky needed
            "ell_": [float(v) for v in gp.ell_],
            "jitter_": [float(v) for v in gp.jitter_],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                # not used by the fit, but unpickling imports response_surrogate
                "scikit-learn": sklearn.__version__,
            },
        }
