"""Fixed Gauss-Legendre grid in t = log(E / E_scale).

The nodes do not depend on the population parameters, so the conditional
log-likelihood table built from them is valid for every proposal. That is what
removes the forward model from the sampling loop.
"""

import numpy as np


class LogEGrid:
    def __init__(self, e_min_Pa, e_max_Pa, n_nodes, e_scale_Pa):
        if not (0.0 < e_min_Pa < e_max_Pa):
            raise ValueError("require 0 < e_min_Pa < e_max_Pa")
        if n_nodes < 8:
            raise ValueError("n_nodes too small to resolve a likelihood spike")
        self.e_min = float(e_min_Pa)
        self.e_max = float(e_max_Pa)
        self.e_scale = float(e_scale_Pa)
        self.n_nodes = int(n_nodes)

        a = np.log(self.e_min / self.e_scale)
        b = np.log(self.e_max / self.e_scale)
        x, w = np.polynomial.legendre.leggauss(self.n_nodes)
        self.t = 0.5 * (b - a) * x + 0.5 * (a + b)
        self.w = 0.5 * (b - a) * w
        self.E = self.e_scale * np.exp(self.t)
        # integrating in t, so dE = E dt supplies the Jacobian
        self.log_weight = np.log(self.w) + self.t + np.log(self.e_scale)

    def spacing_report(self):
        d = np.diff(self.t)
        return {"n_nodes": self.n_nodes,
                "t_span": float(self.t[-1] - self.t[0]),
                "min_spacing": float(d.min()),
                "max_spacing": float(d.max()),
                "mid_spacing": float(d[len(d) // 2])}
