import importlib
import KratosMultiphysics as Kratos

from kratos_forward_model import KratosForwardModel
from likelihood import Likelihood, Prior


class BayesianAnalysis:
    """Analysis stage for Bayesian parameter inference, shaped like a Kratos
    AnalysisStage: Initialize / RunSolutionLoop / Finalize, driven by Parameters."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "problem_data"     : {
                "problem_name"  : "bayesian_inference",
                "echo_level"    : 1
            },
            "forward_model"    : {},
            "parameters"       : [],
            "likelihood"       : {},
            "sampler_settings" : {
                "python_module" : "smc_acs_sampler",
                "Parameters"    : {}
            },
            "output_processes" : []
        }""")

    def __init__(self, model, project_parameters):
        project_parameters.ValidateAndAssignDefaults(self.GetDefaultParameters())
        project_parameters["problem_data"].ValidateAndAssignDefaults(
            self.GetDefaultParameters()["problem_data"])
        self.model = model
        self.settings = project_parameters
        self.echo_level = project_parameters["problem_data"]["echo_level"].GetInt()

        self.forward_model = None
        self.likelihood = None
        self.prior = None
        self.sampler = None
        self.output_processes = []

    # ------------------------------------------------------------------ stages
    def Initialize(self):
        entries = [self.settings["parameters"][i]
                   for i in range(self.settings["parameters"].size())]
        if not entries:
            raise RuntimeError("'parameters' is empty: nothing to infer")

        self.forward_model = KratosForwardModel(
            self.model, self.settings["forward_model"], entries)
        self.likelihood = Likelihood(self.forward_model, self.settings["likelihood"])
        self.prior = Prior(entries)

        sampler_settings = self.settings["sampler_settings"]
        module = importlib.import_module(
            "samplers." + sampler_settings["python_module"].GetString())
        self.sampler = module.Factory(self.likelihood, self.prior,
                                      sampler_settings["Parameters"])

        for i in range(self.settings["output_processes"].size()):
            entry = self.settings["output_processes"][i]
            module = importlib.import_module(
                "processes." + entry["python_module"].GetString())
            self.output_processes.append(
                module.Factory(self.forward_model, self.likelihood, entry["Parameters"]))

    def RunSolutionLoop(self):
        Kratos.Logger.PrintInfo("BayesianAnalysis", "sampling started")
        self.sampler.Run()
        Kratos.Logger.PrintInfo(
            "BayesianAnalysis", "sampling finished: %d levels, %d forward solves, logcE = %.3f"
            % (len(self.sampler.q), self.forward_model.n_solves, self.sampler.Evidence()))

    def Finalize(self):
        for process in self.output_processes:
            if hasattr(process, "ExecuteFinalize"):
                try:
                    process.ExecuteFinalize(self.sampler.levels, self.sampler.q,
                                            self.sampler.Evidence())
                except TypeError:
                    process.ExecuteFinalize(self.sampler.levels, self.sampler.q)
        self.forward_model.Finalize()

    def Run(self):
        self.Initialize()
        self.RunSolutionLoop()
        self.Finalize()
