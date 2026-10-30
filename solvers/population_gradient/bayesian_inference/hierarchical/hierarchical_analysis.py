import importlib
import json
import os
import sys

import numpy as np
import KratosMultiphysics as Kratos

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from likelihood import Prior
from population_likelihood import PopulationLikelihood


class HierarchicalAnalysis:
    """Analysis stage for population inference. Shaped like BayesianAnalysis,
    but there is no forward model: the surrogate and the quadrature table stand
    in for it, so RunSolutionLoop performs zero FEM solves."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "problem_data"     : {
                "problem_name" : "hierarchical_inference",
                "echo_level"   : 1
            },
            "parameters"       : [],
            "likelihood"       : {},
            "sampler_settings" : {
                "python_module" : "smc_acs_sampler",
                "Parameters"    : {}
            },
            "output"           : {
                "output_path" : "output_hierarchical",
                "write_npz"   : true
            }
        }""")

    def __init__(self, model, project_parameters):
        project_parameters.ValidateAndAssignDefaults(self.GetDefaultParameters())
        project_parameters["problem_data"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["problem_data"])
        project_parameters["output"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["output"])
        self.model = model
        self.settings = project_parameters
        self.likelihood = None
        self.prior = None
        self.sampler = None

    def Initialize(self):
        entries = [self.settings["parameters"][i]
                   for i in range(self.settings["parameters"].size())]
        if not entries:
            raise RuntimeError("'parameters' is empty: nothing to infer")

        self.likelihood = PopulationLikelihood(self.settings["likelihood"])
        if len(entries) != self.likelihood.population.n_params:
            raise RuntimeError(
                f"population family '{self.likelihood.family}' needs "
                f"{self.likelihood.population.n_params} parameters "
                f"{self.likelihood.population.names}, but 'parameters' has "
                f"{len(entries)}")
        self.prior = Prior(entries)
        self.param_names = [e["name"].GetString() for e in entries]

        s = self.settings["sampler_settings"]
        module = importlib.import_module("samplers." + s["python_module"].GetString())
        self.sampler = module.Factory(self.likelihood, self.prior, s["Parameters"])

        Kratos.Logger.PrintInfo("HierarchicalAnalysis",
                                json.dumps(self.likelihood.Describe()["quadrature"]))
        d = self.likelihood.Describe()
        Kratos.Logger.PrintInfo(
            "HierarchicalAnalysis",
            "%d specimens, %d sensor(s), table %s = %.2f MiB, sigma = %.6e"
            % (d["n_observations"], d["n_sensors"], d["table_shape"],
               d["table_MiB"], d["sigma_assumed_m"]))

    def _particles(self, level):
        """SMC_aCS hands back each level as (dim, N); smc_acs_sampler reshapes it
        to (-1, dim), which interleaves the parameters once dim > 1. The reshape
        preserves the flat buffer, so the true pairing is recovered here rather
        than by editing the validated sampler."""
        a = np.asarray(level, float).ravel()
        dim = self.prior.dim
        if a.size % dim:
            raise RuntimeError(f"level of size {a.size} is not a multiple of {dim}")
        return np.ascontiguousarray(a.reshape(dim, -1).T)

    def RunSolutionLoop(self):
        Kratos.Logger.PrintInfo("HierarchicalAnalysis", "sampling started")
        self.sampler.Run()
        self.levels = [self._particles(x) for x in self.sampler.levels]
        Kratos.Logger.PrintInfo(
            "HierarchicalAnalysis",
            "finished: %d levels, %d likelihood calls, 0 forward solves, logcE = %.4f"
            % (len(self.sampler.q), self.likelihood.n_calls, self.sampler.Evidence()))

    def Finalize(self):
        out = self.settings["output"]
        path = out["output_path"].GetString()
        os.makedirs(path, exist_ok=True)
        post = self.levels[-1]

        stats = {}
        for k, name in enumerate(self.param_names):
            v = post[:, k]
            stats[name] = {
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            }
        mean_eta = post.mean(axis=0)
        m, s = self.likelihood.population.moments(mean_eta)

        summary = {
            "population_family": self.likelihood.family,
            "parameters": self.param_names,
            "posterior": stats,
            "implied_E_mean_GPa": m / 1e9,
            "implied_E_sd_GPa": s / 1e9,
            "logcE": float(self.sampler.Evidence()),
            "tempering_q": [float(x) for x in np.asarray(self.sampler.q).ravel()],
            "n_levels": len(self.sampler.q),
            "n_likelihood_calls": int(self.likelihood.n_calls),
            "n_forward_solves": 0,
            "data": self.likelihood.Describe(),
        }
        with open(os.path.join(path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if out["write_npz"].GetBool():
            np.savez(os.path.join(path, "posterior.npz"),
                     posterior=post, names=np.array(self.param_names),
                     q=np.asarray(self.sampler.q),
                     **{f"level_{i}": l for i, l in enumerate(self.levels)})

        print("\nposterior")
        for name in self.param_names:
            st = stats[name]
            print("  %-12s mean %14.6e  sd %12.6e  95%% [%.6e, %.6e]"
                  % (name, st["mean"], st["sd"], st["ci95"][0], st["ci95"][1]))
        print("  implied E population: mean %.4f GPa, sd %.4f GPa" % (m / 1e9, s / 1e9))
        print("  logcE = %.4f, %d levels, %d likelihood calls, 0 forward solves"
              % (summary["logcE"], summary["n_levels"], summary["n_likelihood_calls"]))
        print("-> %s" % os.path.join(path, "summary.json"))

    def Run(self):
        self.Initialize()
        self.RunSolutionLoop()
        self.Finalize()
