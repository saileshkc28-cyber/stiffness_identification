import os
import sys

# this package lives in a subfolder; the shared modules (likelihood, samplers,
# response_surrogate) stay where they are, one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import KratosMultiphysics as Kratos

from hierarchical_analysis import HierarchicalAnalysis

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "HierarchicalParameters.json"
    with open(config, "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    model = Kratos.Model()
    HierarchicalAnalysis(model, parameters).Run()
