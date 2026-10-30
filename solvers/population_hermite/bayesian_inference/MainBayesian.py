import sys

import KratosMultiphysics as Kratos
from bayesian_analysis import BayesianAnalysis


def _enabled(parameters, block):
    return (parameters.Has(block)
            and parameters[block].Has("enabled")
            and parameters[block]["enabled"].GetBool())


if __name__ == "__main__":

    # optional config path as the first argument; anything starting with -- is one of
    # the three-point flags (--dry-run, --sensor-file, ...), read later from sys.argv
    config = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
              else "BayesianParameters.json")
    with open(config, "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    three_point = _enabled(parameters, "three_point_inference")
    batch = _enabled(parameters, "batch_inference")

    if three_point and batch:
        raise RuntimeError(
            "three_point_inference.enabled and batch_inference.enabled are both true; "
            "these are different experiments, so disable one of them explicitly "
            "rather than letting the program choose")

    if three_point:
        # three Gauss-Hermite support points, one independent inversion each
        from three_point_bayesian_analysis import RunThreePoint
        RunThreePoint(parameters)

    elif batch:
        # one independent inversion per Phase 1 realization
        from batch_bayesian_analysis import RunBatch
        RunBatch(parameters)

    else:
        # the original single-dataset path, unchanged. BayesianAnalysis validates
        # against its own defaults, which know nothing about the controller blocks
        for block in ("three_point_inference", "batch_inference"):
            if parameters.Has(block):
                parameters.RemoveValue(block)

        model = Kratos.Model()
        analysis = BayesianAnalysis(model, parameters)
        analysis.Run()
