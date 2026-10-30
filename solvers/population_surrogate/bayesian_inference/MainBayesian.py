import KratosMultiphysics as Kratos
from bayesian_analysis import BayesianAnalysis


if __name__ == "__main__":

    with open("BayesianParameters.json", "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    # batch mode runs one independent inversion per Phase 1 realization; absent or
    # disabled, everything below is the original single-dataset path
    batch_enabled = (parameters.Has("batch_inference")
                     and parameters["batch_inference"].Has("enabled")
                     and parameters["batch_inference"]["enabled"].GetBool())

    if batch_enabled:
        from batch_bayesian_analysis import RunBatch
        RunBatch(parameters)
    else:
        # BayesianAnalysis validates against its own defaults, which do not include
        # the batch block, so it has to be dropped before the single run
        if parameters.Has("batch_inference"):
            parameters.RemoveValue("batch_inference")

        model = Kratos.Model()
        analysis = BayesianAnalysis(model, parameters)
        analysis.Run()
