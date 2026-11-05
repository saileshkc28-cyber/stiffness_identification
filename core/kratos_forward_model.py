import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

import sensors as sensor_tools


class KratosForwardModel:
    """Build once, evaluate many. Initialize() costs ~1000x a single solve."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "primal_parameters_file" : "PrimalParametersBayes.json",
            "sensor_data_file"       : "../sensor_placement/sensor_data.json"
        }""")

    def __init__(self, model, settings, parameter_entries):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters())
        self.model = model

        with open(settings["primal_parameters_file"].GetString(), "r") as file_input:
            primal = Kratos.Parameters(file_input.read())

        self.analysis = StructuralMechanicsAnalysis(model, primal)
        self.analysis.Initialize()
        self.root = model[primal["solver_settings"]["model_part_name"].GetString()]
        self.t0 = primal["problem_data"]["start_time"].GetDouble()

        elem_at = {e.Id: i for i, e in enumerate(self.root.Elements)}
        self.zones = []
        for entry in parameter_entries:
            part = model[entry["target_sub_model_part"].GetString()]
            props = list({e.Properties.Id: e.Properties for e in part.Elements}.values())
            variable = Kratos.KratosGlobals.GetVariable(entry["control_variable"].GetString())
            # reference = the value the materials file assigned this zone, read once
            # before any sample overwrites it; JSON may override it explicitly
            ref = (entry["reference_value"].GetDouble()
                   if entry.Has("reference_value") else props[0][variable])
            Kratos.Logger.PrintInfo("KratosForwardModel", "%s: %s (%d elements), reference = %.6e"
                                    % (entry["name"].GetString(),
                                       entry["target_sub_model_part"].GetString(),
                                       len(part.Elements), ref))
            self.zones.append((variable, float(ref), props,
                               np.array([elem_at[e.Id] for e in part.Elements])))

        self.refs = np.array([z[1] for z in self.zones])
        self.prop_ids = [[p.Id for p in z[2]] for z in self.zones]

        self.located = sensor_tools.locate(
            self.root, sensor_tools.read_sensors(settings["sensor_data_file"].GetString()))
        self.nodes = np.array([[n.X0, n.Y0, n.Z0] for n in self.root.Nodes])
        idx = {n.Id: i for i, n in enumerate(self.root.Nodes)}
        self.cells = np.array([[idx[n.Id] for n in e.GetGeometry()]
                               for e in self.root.Elements])
        self.field = None
        self.n_solves = 0

    def Evaluate(self, theta):
        for a, (variable, ref, props, _) in zip(np.atleast_1d(theta), self.zones):
            for p in props:
                p.SetValue(variable, float(a) * ref)

        # Clear() forces a clean rebuild of DOFs/LHS/RHS; without it state leaks between samples
        solver = self.analysis._GetSolver()
        solver.Clear()
        info = solver.GetComputingModelPart().ProcessInfo
        info[Kratos.STEP] = 0
        info[Kratos.TIME] = self.t0
        self.analysis.time = self.t0
        self.analysis.RunSolutionLoop()

        self.field = sensor_tools.displacements(self.root)
        self.n_solves += 1
        return sensor_tools.read(self.field, self.located)

    def YoungsModulus(self, theta):
        E = np.zeros(len(self.cells))
        for a, (_, ref, _, elems) in zip(np.atleast_1d(theta), self.zones):
            E[elems] = float(a) * ref
        return E

    def Finalize(self):
        self.analysis.Finalize()
