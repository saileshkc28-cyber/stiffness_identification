import json
import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from sensors import read_sensors, interpolate

# fixed noise floor, independent of the sensor set: 2% of the undamaged tip deflection
NOISE_SIGMA = 3.77e-08
SEED = 20260802


class CustomStructuralMechanicsAnalysis(StructuralMechanicsAnalysis):
    """Copies YOUNG_MODULUS from Properties onto element data for the VTK block."""

    def FinalizeSolutionStep(self):

        for element in self._GetSolver().GetComputingModelPart().Elements:
            element.SetValue(Kratos.YOUNG_MODULUS, element.Properties[Kratos.YOUNG_MODULUS])

        super().FinalizeSolutionStep()


if __name__ == "__main__":

    model = Kratos.Model()

    with open("PrimalParametersBeam.json", "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    if parameters["output_processes"].Has("sensor_output"):
        parameters["output_processes"].RemoveValue("sensor_output")

    analysis = CustomStructuralMechanicsAnalysis(model, parameters)
    analysis.Run()

    # the planted truth is whatever StructuralMaterials.json assigned, read back per zone
    E_true = {}
    for element in model["Structure"].Elements:
        E_true[element.Properties.Id] = element.Properties[Kratos.YOUNG_MODULUS]

    sensors = read_sensors("../sensor_placement/sensor_data.json")
    u_true = interpolate(model["Structure"], sensors)

    sigma = NOISE_SIGMA
    u_hat = u_true + np.random.default_rng(SEED).normal(0.0, sigma, u_true.shape)

    with open("measured_data.csv", "w") as f:
        f.write("#,type,name,location_0,location_1,location_2,value\n")
        for i, (s, v) in enumerate(zip(sensors, u_hat), 1):
            f.write(f"{i},{s['type']},{s['name']},{s['location'][0]},{s['location'][1]},"
                    f"{s['location'][2]},{v:.16e}\n")

    with open("noise_model.json", "w") as f:
        json.dump({"sigma": sigma, "seed": SEED,
                   "u_true": u_true.tolist(), "u_hat": u_hat.tolist()}, f, indent=2)

    for pid, E in sorted(E_true.items()):
        print(f"\nproperty {pid}: E_true = {E:.6e} Pa")
    print(f"sigma = {sigma:.6e}")
    for s, ut, uh in zip(sensors, u_true, u_hat):
        print(f"{s['name']}: u_true = {ut: .6e}   u_hat = {uh: .6e}")