import os
import sys
import numpy as np
import KratosMultiphysics as Kratos


def Factory(likelihood, prior, settings):
    return SMCaCSSampler(likelihood, prior, settings)


class SMCaCSSampler:
    """Adapter around the unmodified ERA SMC_aCS. The ERA code stays a cited
    black box; per-level statistics are recovered afterwards from the likelihood
    cache, using the particle populations SMC_aCS returns."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module" : "smc_acs_sampler",
            "Parameters"    : {
                "n_particles" : 1000,
                "p"           : 1,
                "burn"        : 2,
                "target_cov"  : 1.5,
                "random_seed" : 0
            }
        }""")

    def __init__(self, likelihood, prior, settings):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters()["Parameters"])
        self.likelihood = likelihood
        self.prior = prior
        self.settings = settings
        self.levels, self.q, self.logcE = [], None, None

    def Run(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "era"))
        from era.SMC_aCS import SMC_aCS

        np.random.seed(self.settings["random_seed"].GetInt())   # ERA uses global state
        _, samplesX, q, logcE = SMC_aCS(self.settings["n_particles"].GetInt(),
                                        self.settings["p"].GetInt(),
                                        self.likelihood,
                                        self.prior.EraObject(),
                                        self.settings["burn"].GetInt(),
                                        self.settings["target_cov"].GetDouble())

        self.levels = [np.atleast_2d(np.asarray(x, float)).reshape(-1, self.prior.dim)
                       for x in samplesX]
        self.q = np.asarray(q)
        self.logcE = float(logcE)
        return self.levels

    def Posterior(self):
        return self.levels[-1]

    def Evidence(self):
        return self.logcE
