"""SMC_aCS with a robust tempering step, for very informative likelihoods.

Each level, ERA's SMC_aCS chooses the tempering step dq by solving
ESS(dq) = target with scipy.optimize.root, starting at dq = 1. ESS falls from N to 1
as dq grows. When the likelihood is very sharp (a hierarchical likelihood over many
specimens), ESS is already flat at 1 near dq = 1, the root search stops where it
started, and the sampler jumps from the prior straight to q = 1 with no tempering.

This module leaves the ERA file untouched. While SMC_aCS runs, the name "optimize"
inside the ERA module points to a stand-in whose root() solves the same equation by
bisection on log(dq), which cannot stall. Everything else in SMC_aCS is unchanged, and
the zone order of the returned particles is fixed as in smc_acs_multizone_sampler.
"""
from types import SimpleNamespace

import numpy as np
from scipy import optimize as scipy_optimize

from samplers.smc_acs_multizone_sampler import SMCaCSMultiZoneSampler


def Factory(likelihood, prior, settings):
    return SMCaCSRobustSampler(likelihood, prior, settings)


class _BisectionRoot:
    """Only what SMC_aCS uses from scipy.optimize: root(), and fsolve() as fallback."""

    @staticmethod
    def root(fun, x0=1.0, method=None):
        def f(log_dq):
            return float(np.atleast_1d(fun(np.exp(log_dq)))[0])

        lo, hi = np.log(1e-14), np.log(10.0)
        if f(hi) > 0.0:            # even a step of 10 keeps the ESS above target
            return SimpleNamespace(success=True, x=np.array([np.exp(hi)]))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return SimpleNamespace(success=True, x=np.array([np.exp(0.5 * (lo + hi))]))

    fsolve = staticmethod(scipy_optimize.fsolve)


class SMCaCSRobustSampler(SMCaCSMultiZoneSampler):
    def Run(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "era"))
        import era.SMC_aCS as era_smc
        original = era_smc.optimize
        era_smc.optimize = _BisectionRoot
        try:
            return super().Run()
        finally:
            era_smc.optimize = original
