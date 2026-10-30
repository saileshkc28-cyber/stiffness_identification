import KratosMultiphysics as Kratos
from bayesian_analysis import BayesianAnalysis


if __name__ == "__main__":

    model = Kratos.Model()

    with open("BayesianParameters.json", "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    analysis = BayesianAnalysis(model, parameters)
    analysis.Run()
