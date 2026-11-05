import csv
import numpy as np
import KratosMultiphysics as Kratos

from era.Distributions.ERADist import ERADist
from era.Distributions.ERANataf import ERANataf


class Likelihood:
    """Likelihood ONLY. SMC_aCS holds the prior in its ERADist object, so adding
    log_prior here would double-count it."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "measured_data_file" : "../damaged_system/measured_data.csv",
            "noise_model"        : {
                "type"  : "gaussian_iid",
                "sigma" : 0.0
            }
        }""")

    def __init__(self, forward_model, settings):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters())
        settings["noise_model"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["noise_model"])

        noise_type = settings["noise_model"]["type"].GetString()
        if noise_type != "gaussian_iid":
            raise RuntimeError(f"unsupported noise model '{noise_type}'")

        self.model = forward_model
        self.sigma = settings["noise_model"]["sigma"].GetDouble()
        with open(settings["measured_data_file"].GetString()) as f:
            self.u_hat = np.array([float(row["value"]) for row in csv.DictReader(f)])
        self.cache = {}

    def Forward(self, theta):
        key = tuple(np.round(np.atleast_1d(theta).astype(float), 12))
        if key not in self.cache:
            u = self.model.Evaluate(theta)
            self.cache[key] = (u, self.model.field.copy())
        return self.cache[key]

    def Cached(self, theta):
        return self.cache.get(tuple(np.round(np.atleast_1d(theta).astype(float), 12)))

    def __call__(self, theta):
        """Scalar float: numpy >= 2 rejects length-1 arrays inside SMC_aCS."""
        r = self.u_hat - self.Forward(theta)[0]
        return float(-0.5 * float(r @ r) / self.sigma**2)


class Prior:
    """Built from the 'parameters' block: one marginal per inferred quantity."""

    def __init__(self, parameter_entries):
        self.marginals = []
        for entry in parameter_entries:
            prior = entry["prior"]
            self.marginals.append(ERADist(prior["type"].GetString(), "PAR",
                                          list(prior["parameters"].GetVector())))
        self.dim = len(self.marginals)

    def EraObject(self):
        """SMC_aCS takes a single ERADist in 1D, an ERANataf otherwise."""
        if self.dim == 1:
            return self.marginals[0]
        return ERANataf(self.marginals, np.identity(self.dim))
